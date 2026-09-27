//! DeBERTa-v2/v3 relative attention without padding or quadratic score storage.
use super::*;

extern "C" {
    fn candle_fa4_deberta_v1(
        dtype: i32,
        device: i32,
        stream: *mut c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        o: *mut c_void,
        offsets: *mut c_void,
        c2p: *mut c_void,
        p2c: *mut c_void,
        buckets: *mut c_void,
        total: i64,
        batch: i64,
        heads: i64,
        span: i64,
        lut_len: i64,
    ) -> i32;
}

/// Prevalidated relative indices for contiguous self-attention positions.
/// A linear lookup replaces a [length,length] relative-position matrix.
#[derive(Clone)]
pub struct RelativeBuckets {
    ids: Tensor,
    span: usize,
    max_len: usize,
}
impl RelativeBuckets {
    pub fn new(
        max_len: usize,
        span: usize,
        logarithmic: bool,
        max_position: usize,
        device: &candle::Device,
    ) -> Result<Self> {
        if max_len == 0
            || max_len > (i32::MAX as usize) / 2
            || span == 0
            || span > (i32::MAX as usize) / 2
            || (logarithmic && (span < 4 || max_position <= span / 2 + 1))
        {
            candle::bail!("invalid DeBERTa relative position configuration")
        }
        let mid = (span / 2) as f32;
        let ids: Vec<u32> = (1 - max_len as i64..max_len as i64)
            .map(|r| {
                let a = r.abs() as f32;
                let bucket = if logarithmic && a > mid {
                    let log = ((a / mid).ln() / (((max_position - 1) as f32) / mid).ln()
                        * (mid - 1.))
                        .ceil()
                        + mid;
                    (log as i64) * r.signum()
                } else {
                    r
                };
                (bucket + span as i64).clamp(0, (2 * span - 1) as i64) as u32
            })
            .collect();
        Ok(Self {
            ids: Tensor::new(ids.as_slice(), device)?,
            span,
            max_len,
        })
    }
}

/// Q,K,V are contiguous [total,heads,64]. K must already contain the dtype-rounded
/// division by sqrt(head_dim * scale_factor). c2p and p2c are contiguous
/// [heads,total,2*span], each scaled after its dtype-rounded matrix product.
/// For local positions i,j the two table entries are c2p[h,i,bucket(i-j)] and
/// p2c[h,j,bucket(i-j)]. Missing terms can be represented by zero tables.
/// The FA4 score hook preserves model-dtype score-addition rounding. Online
/// softmax may differ numerically from eager attention; no bitwise claim is made.
/// Inference only, self-attention only; no cross-sequence attention or padding.
pub fn deberta_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    c2p: &Tensor,
    p2c: &Tensor,
    lengths: &Seqlens,
    buckets: &RelativeBuckets,
) -> Result<Tensor> {
    let (total, heads, dim) = q.dims3()?;
    if total != lengths.total
        || heads == 0
        || dim != 64
        || buckets.max_len < lengths.max_len
        || !matches!(q.dtype(), DType::F16 | DType::BF16)
        || !q.device().is_cuda()
    {
        candle::bail!("DeBERTa FA4 requires packed FP16/BF16 CUDA d64 self-attention")
    }
    for t in [q, k, v, c2p, p2c] {
        if !t.is_contiguous()
            || !t.device().same_device(q.device())
            || t.dtype() != q.dtype()
            || t.elem_count() > i32::MAX as usize
        {
            candle::bail!(
                "DeBERTa tensors must be contiguous, same-device/dtype, with int32-sized storage"
            )
        }
    }
    if k.dims() != q.dims()
        || v.dims() != q.dims()
        || c2p.dims() != [heads, total, 2 * buckets.span]
        || p2c.dims() != c2p.dims()
        || !lengths.offsets.device().same_device(q.device())
        || !buckets.ids.device().same_device(q.device())
        || [q, k, v]
            .iter()
            .any(|t| !t.layout().start_offset().is_multiple_of(8))
    {
        candle::bail!("invalid DeBERTa shapes, alignment, or metadata device")
    }
    q.apply_op3_no_bwd(
        k,
        v,
        &Deberta {
            c2p: c2p.clone(),
            p2c: p2c.clone(),
            lengths: lengths.clone(),
            buckets: buckets.clone(),
        },
    )
}
struct Deberta {
    c2p: Tensor,
    p2c: Tensor,
    lengths: Seqlens,
    buckets: RelativeBuckets,
}
impl CustomOp3 for Deberta {
    fn name(&self) -> &'static str {
        "fa4-deberta-varlen"
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
        candle::bail!("DeBERTa FA4 requires CUDA")
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
            DType::F16 => self.launch::<half::f16>(q, ql, k, kl, v, vl, 0),
            DType::BF16 => self.launch::<half::bf16>(q, ql, k, kl, v, vl, 1),
            _ => candle::bail!("unsupported DeBERTa dtype"),
        }
    }
}
impl Deberta {
    #[allow(clippy::too_many_arguments)]
    fn launch<
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
            candle::bail!("DeBERTa AOT bundle currently requires SM90")
        }
        let shape = ql.shape().clone();
        let q = q.as_cuda_slice::<T>()?.slice(ql.start_offset()..);
        let k = k.as_cuda_slice::<T>()?.slice(kl.start_offset()..);
        let v = v.as_cuda_slice::<T>()?.slice(vl.start_offset()..);
        let (as_, al) = self.c2p.storage_and_layout();
        let (bs, bl) = self.p2c.storage_and_layout();
        let (cs, cl) = self.lengths.offsets.storage_and_layout();
        let (ls, ll) = self.buckets.ids.storage_and_layout();
        let (Storage::Cuda(as_), Storage::Cuda(bs), Storage::Cuda(cs), Storage::Cuda(ls)) =
            (&*as_, &*bs, &*cs, &*ls)
        else {
            candle::bail!("CUDA metadata required")
        };
        let a = as_.as_cuda_slice::<T>()?.slice(al.start_offset()..);
        let b = bs.as_cuda_slice::<T>()?.slice(bl.start_offset()..);
        let c = cs.as_cuda_slice::<u32>()?.slice(cl.start_offset()..);
        let l = ls.as_cuda_slice::<u32>()?.slice(ll.start_offset()..);
        let mut out = unsafe { dev.alloc::<T>(shape.elem_count()) }?;
        let (qp, _qg) = q.device_ptr(&stream);
        let (kp, _kg) = k.device_ptr(&stream);
        let (vp, _vg) = v.device_ptr(&stream);
        let (ap, _ag) = a.device_ptr(&stream);
        let (bp, _bg) = b.device_ptr(&stream);
        let (cp, _cg) = c.device_ptr(&stream);
        let (lp, _lg) = l.device_ptr(&stream);
        {
            let (op, _og) = out.device_ptr_mut(&stream);
            let rc = unsafe {
                candle_fa4_deberta_v1(
                    dtype,
                    stream.context().ordinal() as i32,
                    stream.cu_stream() as *mut c_void,
                    qp as *mut c_void,
                    kp as *mut c_void,
                    vp as *mut c_void,
                    op as *mut c_void,
                    cp as *mut c_void,
                    ap as *mut c_void,
                    bp as *mut c_void,
                    lp as *mut c_void,
                    shape.dims()[0] as i64,
                    (self.lengths.offsets.elem_count() - 1) as i64,
                    shape.dims()[1] as i64,
                    self.buckets.span as i64,
                    self.buckets.ids.elem_count() as i64,
                )
            };
            if rc != 0 {
                let e = unsafe { CStr::from_ptr(candle_fa4_error_v4()) }.to_string_lossy();
                candle::bail!("DeBERTa FA4 launch failed: {e}")
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev.clone()), shape))
    }
}
