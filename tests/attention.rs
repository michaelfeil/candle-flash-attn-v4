use candle::{DType, Device, Result, Tensor};
use candle_flash_attn_v4::{flash_attn_varlen, Mask, Seqlens};

#[test]
fn packed_masks_match_uniform_attention() -> Result<()> {
    let dev = Device::new_cuda_with_stream(0)?;
    let offsets = [0u32, 1, 8, 73, 202];
    let lengths = Seqlens::new(&offsets, &dev)?;
    for (h, hk, d, mask) in [
        (16, 16, 64, Mask::Global),
        (12, 12, 64, Mask::Global),
        (12, 12, 64, Mask::Local64),
        (32, 8, 128, Mask::Causal),
    ] {
        let total = 202;
        let q = Tensor::zeros((total, h, d), DType::F16, &dev)?;
        let k = Tensor::zeros((total, hk, d), DType::F16, &dev)?;
        let values: Vec<f32> = (0..total * hk * d)
            .map(|i| ((i / (hk * d)) % 17) as f32 / 16.)
            .collect();
        let v = Tensor::from_vec(values, (total, hk, d), &dev)?.to_dtype(DType::F16)?;
        let actual = flash_attn_varlen(&q, &k, &v, &lengths, mask)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for pair in offsets.windows(2) {
            let (start, end) = (pair[0] as usize, pair[1] as usize);
            for token in start..end {
                let (lo, hi) = match mask {
                    Mask::Global => (start, end),
                    Mask::Causal => (start, token + 1),
                    Mask::Local64 => (start.max(token.saturating_sub(64)), end.min(token + 65)),
                };
                let expected =
                    (lo..hi).map(|i| (i % 17) as f32 / 16.).sum::<f32>() / (hi - lo) as f32;
                for &value in &actual[token * h * d..(token + 1) * h * d] {
                    assert!(
                        (value - expected).abs() < 0.002,
                        "{mask:?}: {value} vs {expected}"
                    );
                }
            }
        }
        let wrong = Seqlens::new(&[0, 201], &dev)?;
        assert!(flash_attn_varlen(&q, &k, &v, &wrong, mask).is_err());
    }
    for offsets in [&[1u32, 2][..], &[0, 0], &[0, 2, 1], &[0, u32::MAX], &[0]] {
        assert!(Seqlens::new(offsets, &dev).is_err());
    }
    Ok(())
}
