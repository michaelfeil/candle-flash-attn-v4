//! Experimental AOT FA4 bridge, limited to the explicitly exported FP16/BF16 kernels.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Storage, Tensor};
use std::ffi::{c_char, c_void, CStr};

extern "C" {
    fn candle_fa4_forward_v4(
        mode: i32,
        dtype: i32,
        device: i32,
        stream: *mut c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        o: *mut c_void,
        offsets: *mut c_void,
        offsets_k: *mut c_void,
        total: i64,
        total_k: i64,
        batch: i64,
        heads: i64,
        kv_heads: i64,
        scale: f64,
        left: i32,
        right: i32,
        strides: *const i64,
    ) -> i32;
    fn candle_fa4_error_v4() -> *const c_char;
}

/// Validated packed sequence boundaries, uploaded once to CUDA.
/// Empty sequences are currently rejected. Boundaries are uploaded at construction.
#[derive(Clone)]
pub struct Seqlens {
    offsets: Tensor,
    total: usize,
    max_len: usize,
}
impl Seqlens {
    pub fn new(offsets: &[u32], device: &candle::Device) -> Result<Self> {
        if offsets.len() < 2
            || offsets[0] != 0
            || offsets.windows(2).any(|v| v[0] >= v[1])
            || *offsets.last().unwrap() > i32::MAX as u32
            || offsets.len() > i32::MAX as usize
        {
            candle::bail!("offsets must start at zero, strictly increase, and fit signed int32")
        }
        if !device.is_cuda() {
            candle::bail!("FA4 requires CUDA")
        }
        Ok(Self {
            offsets: Tensor::new(offsets, device)?,
            total: *offsets.last().unwrap() as usize,
            max_len: offsets
                .windows(2)
                .map(|w| (w[1] - w[0]) as usize)
                .max()
                .unwrap(),
        })
    }
}

/// Available masks in the initial AOT bundle.
#[derive(Clone, Copy, Debug)]
pub enum Mask {
    Global,
    Causal,
    Local64,
    /// Inclusive distances to the left and right of each query position.
    Window {
        left: usize,
        right: usize,
    },
}

/// Runtime attention parameters. None selects the conventional inverse-square-root scale.
#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub mask: Mask,
    pub softmax_scale: Option<f32>,
}
impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            mask: Mask::Global,
            softmax_scale: None,
        }
    }
}

/// Packed self-attention, Q=[tokens, heads, dim], K/V=[tokens, kv_heads, dim].
/// Unsupported configurations return an error; no implicit backend fallback.
/// This inference operation does not provide gradients.
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lengths: &Seqlens,
    mask: Mask,
) -> Result<Tensor> {
    flash_attn_varlen_with_config(
        q,
        k,
        v,
        lengths,
        AttentionConfig {
            mask,
            softmax_scale: None,
        },
    )
}

/// Packed self-attention with explicit scale and mask parameters.
pub fn flash_attn_varlen_with_config(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lengths: &Seqlens,
    config: AttentionConfig,
) -> Result<Tensor> {
    flash_attn_varlen_cross(q, k, v, lengths, lengths, config)
}

/// Packed cross-attention with independent Q/KV lengths and equal batch counts.
/// Causal and local masks align the bottom-right corners: query position i is
/// centered on key position i + kv_length - q_length. Fully masked rows return zero.
pub fn flash_attn_varlen_cross(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_lengths: &Seqlens,
    kv_lengths: &Seqlens,
    config: AttentionConfig,
) -> Result<Tensor> {
    if q.rank() != 3 || q.dims()[0] != q_lengths.total {
        candle::bail!("Q token count does not match sequence boundaries")
    }
    if k.rank() != 3
        || k.dims()[0] != kv_lengths.total
        || q_lengths.offsets.elem_count() != kv_lengths.offsets.elem_count()
    {
        candle::bail!("KV token count or Q/KV batch count does not match boundaries")
    }
    let (causal, left, right) = match config.mask {
        Mask::Global => (false, None, None),
        Mask::Causal => (true, None, None),
        Mask::Local64 => (false, Some(64), Some(64)),
        Mask::Window { left, right } => (false, Some(left), Some(right)),
    };
    if left.into_iter().chain(right).any(|v| v > i32::MAX as usize) {
        candle::bail!("attention window exceeds int32")
    }
    let left = left.map(|v| v.min(q_lengths.max_len.max(kv_lengths.max_len) - 1));
    let right = right.map(|v| v.min(q_lengths.max_len.max(kv_lengths.max_len) - 1));
    let scale = config
        .softmax_scale
        .unwrap_or(((q.dims()[2] as f64).sqrt().recip()) as f32);
    if !scale.is_finite() {
        candle::bail!("attention scale must be finite")
    }
    // The native online softmax scales already-masked (-inf) logits, so its
    // scale must be positive. Preserve nonpositive-scale semantics through Q.
    let adjusted_q;
    let (q, scale) = if scale <= 0.0 {
        adjusted_q = if scale == 0.0 {
            q.zeros_like()?
        } else {
            q.neg()?
        };
        (&adjusted_q, if scale == 0.0 { 1.0 } else { -scale })
    } else {
        (q, scale)
    };
    try_forward(
        q,
        k,
        v,
        &q_lengths.offsets,
        &kv_lengths.offsets,
        scale,
        causal,
        left,
        right,
    )?
    .ok_or_else(|| {
        candle::Error::Msg(
            "unsupported FA4 shape, dtype, layout, or device; see the supported configurations"
                .into(),
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn try_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offsets_q: &Tensor,
    offsets_k: &Tensor,
    scale: f32,
    causal: bool,
    left: Option<usize>,
    right: Option<usize>,
) -> Result<Option<Tensor>> {
    if q.rank() != 3
        || k.rank() != 3
        || v.dims() != k.dims()
        || offsets_q.elem_count() != offsets_k.elem_count()
    {
        return Ok(None);
    }
    let (total, h, d) = q.dims3()?;
    let (kt, hk, kd) = k.dims3()?;
    if total == 0
        || kt == 0
        || d != kd
        || total
            .checked_mul(h * d)
            .is_none_or(|v| v > i32::MAX as usize)
        || h == 0
        || hk == 0
    {
        return Ok(None);
    }
    let mode = match (d, causal, left, right) {
        (64, false, None, None) if h == hk => 0,
        (64, false, Some(_), Some(_)) if h == hk => 1,
        (128, true, None, None) if h % hk == 0 && h / hk == 4 => 2,
        (128, false, None, None) if h % hk == 0 && h / hk == 2 => 3,
        _ => return Ok(None),
    };
    for t in [q, k, v] {
        if !matches!(t.dtype(), DType::F16 | DType::BF16)
            || t.dtype() != q.dtype()
            || !t.device().same_device(q.device())
            || t.stride()[2] != 1
            || t.stride()[1] != t.dims()[2]
            || t.stride()[0] < t.dims()[1] * t.dims()[2]
            || !t.stride()[0].is_multiple_of(8)
            || !t.layout().start_offset().is_multiple_of(8)
            || (t.dims()[0] - 1)
                .checked_mul(t.stride()[0])
                .and_then(|v| v.checked_add(t.dims()[1] * t.dims()[2]))
                .is_none_or(|v| v > i32::MAX as usize)
            || t.stride().iter().any(|&s| s > i32::MAX as usize)
        {
            return Ok(None);
        }
    }
    for offsets in [offsets_q, offsets_k] {
        if offsets.rank() != 1
            || offsets.elem_count() < 2
            || offsets.dtype() != DType::U32
            || !offsets.is_contiguous()
            || !offsets.device().same_device(q.device())
        {
            return Ok(None);
        }
    }
    q.apply_op3_no_bwd(
        k,
        v,
        &Fa4 {
            mode,
            scale,
            left: left.map_or(-1, |v| v as i32),
            right: right.map_or(-1, |v| v as i32),
            offsets: offsets_q.clone(),
            offsets_k: offsets_k.clone(),
        },
    )
    .map(Some)
}

struct Fa4 {
    mode: i32,
    scale: f32,
    left: i32,
    right: i32,
    offsets: Tensor,
    offsets_k: Tensor,
}
impl CustomOp3 for Fa4 {
    fn name(&self) -> &'static str {
        "experimental-fa4-aot"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("FA4 native bridge requires CUDA")
    }
    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F16 => self.cuda_fwd_t::<half::f16>(q, ql, k, kl, v, vl, 0),
            DType::BF16 => self.cuda_fwd_t::<half::bf16>(q, ql, k, kl, v, vl, 1),
            _ => candle::bail!("FA4 requires FP16 or BF16"),
        }
    }
}

impl Fa4 {
    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
        dtype: i32,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = q.device();
        let stream = dev.cuda_stream();
        use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::*;
        let major = stream
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(candle::Error::wrap)?;
        let minor = stream
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(candle::Error::wrap)?;
        if (major, minor) != (9, 0) {
            candle::bail!("this AOT bundle requires SM90; other FA4 targets are not packaged yet")
        }
        let shape = ql.shape().clone();
        let q = q.as_cuda_slice::<T>()?.slice(ql.start_offset()..);
        let k = k.as_cuda_slice::<T>()?.slice(kl.start_offset()..);
        let v = v.as_cuda_slice::<T>()?.slice(vl.start_offset()..);
        let (offset_storage, ol) = self.offsets.storage_and_layout();
        let Storage::Cuda(offset_storage) = &*offset_storage else {
            candle::bail!("FA4 offsets must be CUDA")
        };
        let offsets = offset_storage
            .as_cuda_slice::<u32>()?
            .slice(ol.start_offset()..);
        let (offset_k_storage, okl) = self.offsets_k.storage_and_layout();
        let Storage::Cuda(offset_k_storage) = &*offset_k_storage else {
            candle::bail!("FA4 KV offsets must be CUDA")
        };
        let offsets_k = offset_k_storage
            .as_cuda_slice::<u32>()?
            .slice(okl.start_offset()..);
        let mut output = unsafe { dev.alloc::<T>(shape.elem_count()) }?;
        let out_layout = Layout::contiguous(&shape);
        let strides: Vec<i64> = [ql, kl, vl, &out_layout]
            .into_iter()
            .flat_map(|l| l.stride().iter().map(|&s| s as i64))
            .collect();
        // Keep all pointer guards alive until launch so Candle records stream dependencies.
        let (qp, _q_guard) = q.device_ptr(&stream);
        let (kp, _k_guard) = k.device_ptr(&stream);
        let (vp, _v_guard) = v.device_ptr(&stream);
        let (cp, _c_guard) = offsets.device_ptr(&stream);
        let (ckp, _ck_guard) = offsets_k.device_ptr(&stream);
        {
            let (op, _o_guard) = output.device_ptr_mut(&stream);
            let rc = unsafe {
                candle_fa4_forward_v4(
                    self.mode,
                    dtype,
                    stream.context().ordinal() as i32,
                    stream.cu_stream() as *mut c_void,
                    qp as *mut c_void,
                    kp as *mut c_void,
                    vp as *mut c_void,
                    op as *mut c_void,
                    cp as *mut c_void,
                    ckp as *mut c_void,
                    shape.dims()[0] as i64,
                    kl.shape().dims()[0] as i64,
                    (self.offsets.elem_count() - 1) as i64,
                    shape.dims()[1] as i64,
                    kl.shape().dims()[1] as i64,
                    self.scale as f64,
                    self.left,
                    self.right,
                    strides.as_ptr(),
                )
            };
            if rc != 0 {
                let error = unsafe { CStr::from_ptr(candle_fa4_error_v4()) }.to_string_lossy();
                candle::bail!("FA4 native launch failed: {error}")
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(output, dev.clone()), shape))
    }
}

#[cfg(feature = "deberta")]
mod deberta;
#[cfg(feature = "deberta")]
pub use deberta::{deberta_attn_varlen, RelativeBuckets};
