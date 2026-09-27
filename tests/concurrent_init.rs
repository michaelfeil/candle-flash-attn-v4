use candle::{DType, Device, Result, Tensor};
use candle_flash_attn_v4::{flash_attn_varlen_with_config, AttentionConfig, Mask, Seqlens};
use std::sync::{Arc, Barrier};

// Run this integration binary in a fresh process so no export is preinitialized.
// Use an external timeout: a broken native loader spins rather than returning.
#[test]
fn concurrent_cold_exports_complete_and_preserve_outputs() -> Result<()> {
    let barrier = Arc::new(Barrier::new(8));
    let mut workers = Vec::new();
    for _ in 0..8 {
        let device = Device::new_cuda(0)?;
        let barrier = barrier.clone();
        workers.push(std::thread::spawn(move || -> Result<()> {
            let lengths = Seqlens::new(&[0, 1, 4], &device)?;
            for dtype in [DType::F16, DType::BF16] {
                for mask in [
                    Mask::Global,
                    Mask::Window { left: 1, right: 1 },
                    Mask::Causal,
                ] {
                    let (heads, kv_heads, dim) = match mask {
                        Mask::Causal => (4, 1, 128),
                        _ => (4, 4, 64),
                    };
                    let q = Tensor::zeros((4, heads, dim), dtype, &device)?;
                    let k = Tensor::zeros((4, kv_heads, dim), dtype, &device)?;
                    let v = Tensor::full(0.25f32, (4, kv_heads, dim), &device)?.to_dtype(dtype)?;
                    barrier.wait();
                    let out = flash_attn_varlen_with_config(
                        &q,
                        &k,
                        &v,
                        &lengths,
                        AttentionConfig {
                            mask,
                            softmax_scale: None,
                        },
                    )?;
                    for value in out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()? {
                        assert!((value - 0.25).abs() < 0.001);
                    }
                }
                #[cfg(feature = "deberta")]
                {
                    use candle_flash_attn_v4::{deberta_attn_varlen, RelativeBuckets};
                    let q = Tensor::zeros((4, 4, 64), dtype, &device)?;
                    let v = Tensor::full(0.25f32, (4, 4, 64), &device)?.to_dtype(dtype)?;
                    let rel = Tensor::zeros((4, 4, 8), dtype, &device)?;
                    let buckets = RelativeBuckets::new(3, 4, false, 512, &device)?;
                    barrier.wait();
                    let out = deberta_attn_varlen(&q, &q, &v, &rel, &rel, &lengths, &buckets)?;
                    for value in out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()? {
                        assert!((value - 0.25).abs() < 0.001);
                    }
                }
            }
            Ok(())
        }));
    }
    for worker in workers {
        worker.join().expect("CUDA worker panicked")?;
    }
    Ok(())
}
