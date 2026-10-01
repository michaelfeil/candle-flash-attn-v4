#![cfg(feature = "paged")]
use candle::{DType, Device, Result, Tensor};
use candle_flash_attn_v4::{
    flash_attn_paged, flash_attn_varlen_cross, AttentionConfig, Mask, PagedKv, Seqlens,
};

#[test]
fn paged_matches_packed_with_runtime_geometry() -> Result<()> {
    let dev = Device::new_cuda_with_stream(0)?;
    // Different batch, head counts, page count and lengths from AOT export.
    let tables = vec![vec![11, 3, 7], vec![2], vec![8, 1, 12, 0, 6]];
    let lengths = [151u32, 17, 277];
    let metadata = PagedKv::new(&lengths, &tables, 13, &dev)?;
    let qs = Seqlens::new(&[0, 7, 8, 137], &dev)?;
    let ks = Seqlens::new(&[0, 151, 168, 445], &dev)?;
    let rows: Vec<u32> = tables
        .iter()
        .zip(lengths)
        .flat_map(|(t, n)| (0..n).map(|i| t[i as usize / 64] * 64 + i % 64))
        .collect();
    let rows = Tensor::new(rows.as_slice(), &dev)?;
    for dtype in [DType::F16, DType::BF16] {
        for ratio in [2, 4] {
            let hk = 2;
            let h = hk * ratio;
            let q = Tensor::randn(0f32, 0.3, (137, h, 128), &dev)?.to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 0.3, (13, 64, hk, 128), &dev)?.to_dtype(dtype)?;
            let v = Tensor::randn(0f32, 1., (13, 64, hk, 128), &dev)?.to_dtype(dtype)?;
            let packed_k = k.reshape((13 * 64, hk, 128))?.index_select(&rows, 0)?;
            let packed_v = v.reshape((13 * 64, hk, 128))?.index_select(&rows, 0)?;
            let actual = flash_attn_paged(&q, &k, &v, &qs, &metadata, 1. / 128f32.sqrt())?;
            let expected = flash_attn_varlen_cross(
                &q,
                &packed_k,
                &packed_v,
                &qs,
                &ks,
                AttentionConfig {
                    mask: Mask::Causal,
                    softmax_scale: None,
                },
            )?;
            let delta = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(delta <= 0.002, "{dtype:?} GQA{ratio}: {delta}");
        }
    }
    Ok(())
}
