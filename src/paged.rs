//! Hopper causal d128 GQA with 64-token pages, validated on the host once per batch.
use super::*;
use candle::Device;
pub const PAGE_SIZE: usize = 64;

#[derive(Clone)]
pub struct PagedKv {
    table: Tensor,
    used: Tensor,
    pages: usize,
}
impl PagedKv {
    pub fn new(
        lengths: &[u32],
        tables: &[Vec<u32>],
        pages: usize,
        device: &Device,
    ) -> Result<Self> {
        if lengths.is_empty()
            || lengths.len() != tables.len()
            || pages == 0
            || pages > i32::MAX as usize
        {
            candle::bail!("invalid paged KV batch")
        }
        let columns = tables.iter().map(Vec::len).max().unwrap_or(0);
        let mut packed = vec![0u32; lengths.len() * columns];
        for (i, (&len, table)) in lengths.iter().zip(tables).enumerate() {
            if len == 0
                || len > i32::MAX as u32
                || (len as usize).div_ceil(PAGE_SIZE) > table.len()
                || table.iter().any(|&p| p as usize >= pages)
            {
                candle::bail!("invalid KV page table or sequence length")
            }
            packed[i * columns..i * columns + table.len()].copy_from_slice(table);
        }
        Ok(Self {
            table: Tensor::from_vec(packed, (lengths.len(), columns), device)?,
            used: Tensor::new(lengths, device)?,
            pages,
        })
    }
}

pub fn flash_attn_paged(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lengths: &Seqlens,
    kv: &PagedKv,
    scale: f32,
) -> Result<Tensor> {
    let (tokens, heads, dim) = q.dims3()?;
    let (pages, page_size, kv_heads, kdim) = k.dims4()?;
    if compiled_compute_capability() != 90
        || dim != 128
        || kdim != 128
        || page_size != PAGE_SIZE
        || kv_heads == 0
        || ![2, 4]
            .iter()
            .any(|&ratio| kv_heads.checked_mul(ratio) == Some(heads))
        || tokens != lengths.total
        || pages != kv.pages
        || v.dims() != k.dims()
        || lengths.offsets.elem_count() != kv.used.elem_count() + 1
        || !scale.is_finite()
        || scale <= 0.
    {
        candle::bail!("unsupported paged FA4 geometry")
    }
    if !matches!(q.dtype(), DType::F16 | DType::BF16)
        || !q.device().is_cuda()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || q.stride()[2] != 1
        || q.stride()[1] != 128
        || q.stride()[0] < heads * 128
        || !q.stride()[0].is_multiple_of(8)
    {
        candle::bail!("unsupported paged FA4 dtype or layout")
    }
    for t in [q, k, v] {
        if t.dtype() != q.dtype()
            || !t.device().same_device(q.device())
            || !t.layout().start_offset().is_multiple_of(8)
            || t.elem_count() > i32::MAX as usize
        {
            candle::bail!("paged FA4 tensor mismatch or oversized pool")
        }
    }
    for t in [&lengths.offsets, &kv.table, &kv.used] {
        if !t.device().same_device(q.device()) {
            candle::bail!("paged metadata must share Q's device")
        }
    }
    q.apply_op3_no_bwd(
        k,
        v,
        &Paged {
            mode: if heads / kv_heads == 2 { 0 } else { 1 },
            scale,
            offsets: lengths.offsets.clone(),
            kv: kv.clone(),
        },
    )
}

extern "C" {
    fn candle_fa4_paged_v1(
        mode: i32,
        dtype: i32,
        device: i32,
        stream: *mut c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        o: *mut c_void,
        offsets: *mut c_void,
        used: *mut c_void,
        table: *mut c_void,
        total: i64,
        pages: i64,
        batch: i64,
        columns: i64,
        heads: i64,
        kv_heads: i64,
        scale: f64,
        strides: *const i64,
    ) -> i32;
}
struct Paged {
    mode: i32,
    scale: f32,
    offsets: Tensor,
    kv: PagedKv,
}
impl CustomOp3 for Paged {
    fn name(&self) -> &'static str {
        "fa4-paged"
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
        candle::bail!("paged FA4 requires CUDA")
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
            _ => candle::bail!("paged FA4 requires half tensors"),
        }
    }
}
impl Paged {
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
        if major * 10 + minor != 90 {
            candle::bail!("paged FA4 requires SM90")
        }
        let q = q.as_cuda_slice::<T>()?.slice(ql.start_offset()..);
        let k = k.as_cuda_slice::<T>()?.slice(kl.start_offset()..);
        let v = v.as_cuda_slice::<T>()?.slice(vl.start_offset()..);
        let (os, ol) = self.offsets.storage_and_layout();
        let (us, ul) = self.kv.used.storage_and_layout();
        let (ts, tl) = self.kv.table.storage_and_layout();
        let (Storage::Cuda(os), Storage::Cuda(us), Storage::Cuda(ts)) = (&*os, &*us, &*ts) else {
            candle::bail!("paged FA4 metadata must be CUDA")
        };
        let offsets = os.as_cuda_slice::<u32>()?.slice(ol.start_offset()..);
        let used = us.as_cuda_slice::<u32>()?.slice(ul.start_offset()..);
        let table = ts.as_cuda_slice::<u32>()?.slice(tl.start_offset()..);
        let shape = ql.shape().clone();
        let out_layout = Layout::contiguous(&shape);
        let strides: Vec<i64> = [ql, kl, vl, &out_layout]
            .into_iter()
            .flat_map(|l| l.stride().iter().map(|&v| v as i64))
            .collect();
        let mut output = unsafe { dev.alloc::<T>(shape.elem_count()) }?;
        let (qp, _q) = q.device_ptr(&stream);
        let (kp, _k) = k.device_ptr(&stream);
        let (vp, _v) = v.device_ptr(&stream);
        let (cp, _c) = offsets.device_ptr(&stream);
        let (up, _u) = used.device_ptr(&stream);
        let (tp, _t) = table.device_ptr(&stream);
        {
            let (op, _o) = output.device_ptr_mut(&stream);
            // Validation bounds all page indices and lengths. All pointer guards
            // remain live through the launch to track stream dependencies.
            let rc = unsafe {
                candle_fa4_paged_v1(
                    self.mode,
                    dtype,
                    stream.context().ordinal() as i32,
                    stream.cu_stream() as *mut c_void,
                    qp as *mut c_void,
                    kp as *mut c_void,
                    vp as *mut c_void,
                    op as *mut c_void,
                    cp as *mut c_void,
                    up as *mut c_void,
                    tp as *mut c_void,
                    shape.dims()[0] as i64,
                    kl.dims()[0] as i64,
                    self.kv.used.elem_count() as i64,
                    self.kv.table.dims()[1] as i64,
                    shape.dims()[1] as i64,
                    kl.dims()[2] as i64,
                    self.scale as f64,
                    strides.as_ptr(),
                )
            };
            if rc != 0 {
                let message = unsafe { CStr::from_ptr(candle_fa4_error_v4()) }.to_string_lossy();
                candle::bail!("paged FA4 launch failed: {message}")
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(output, dev.clone()), shape))
    }
}
