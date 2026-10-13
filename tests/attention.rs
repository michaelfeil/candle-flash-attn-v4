use candle::{DType, Device, Result, Tensor};
use candle_flash_attn_v4::{
    flash_attn_varlen, flash_attn_varlen_with_config, AttentionConfig, Mask, Seqlens,
};

#[test]
fn nonuniform_attention_matches_cpu_reference() -> Result<()> {
    let dev = Device::new_cuda_with_stream(0)?;
    let offsets = [0u32, 1, 8, 73, 202];
    let lengths = Seqlens::new(&offsets, &dev)?;
    for scale in [0.37f32, 0.0, -0.25] {
        for dtype in [DType::F16, DType::BF16] {
            for (h, hk, d, mask) in [
                (3, 3, 64, Mask::Global),
                (5, 5, 64, Mask::Window { left: 7, right: 13 }),
                (4, 1, 128, Mask::Causal),
                (16, 16, 64, Mask::Global),
                (12, 12, 64, Mask::Global),
                (12, 12, 64, Mask::Local64),
                (32, 8, 128, Mask::Causal),
                (16, 8, 128, Mask::Causal),
                (6, 3, 128, Mask::Causal),
                (16, 8, 128, Mask::Global),
                (6, 3, 128, Mask::Global),
            ] {
                // Multiples of 1/32 are exactly representable in both input dtypes.
                let qv: Vec<f32> = (0..202 * h * d)
                    .map(|i| ((i * 17 % 67) as f32 - 33.) / 32.)
                    .collect();
                let kv: Vec<f32> = (0..202 * hk * d)
                    .map(|i| ((i * 7 % 53) as f32 - 26.) / 32.)
                    .collect();
                let vv: Vec<f32> = (0..202 * hk * d)
                    .map(|i| ((i * 11 % 71) as f32 - 35.) / 32.)
                    .collect();
                let q = Tensor::from_vec(qv.clone(), (202, h, d), &dev)?.to_dtype(dtype)?;
                let k = Tensor::from_vec(kv.clone(), (202, hk, d), &dev)?.to_dtype(dtype)?;
                let v = Tensor::from_vec(vv.clone(), (202, hk, d), &dev)?.to_dtype(dtype)?;
                let config = AttentionConfig {
                    mask,
                    softmax_scale: Some(scale),
                };
                let actual = flash_attn_varlen_with_config(&q, &k, &v, &lengths, config)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                // Exercise doubled row strides and nonzero storage offsets, as in
                // fused QKV views, without changing the logical input values.
                let view = |t: &Tensor| -> Result<Tensor> {
                    Tensor::stack(&[t, t], 1)?.narrow(1, 1, 1)?.squeeze(1)
                };
                let strided = flash_attn_varlen_with_config(
                    &view(&q)?,
                    &view(&k)?,
                    &view(&v)?,
                    &lengths,
                    config,
                )?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
                assert_eq!(
                    actual, strided,
                    "{dtype:?} {mask:?}: strided input changed output"
                );
                for pair in offsets.windows(2) {
                    let (start, end) = (pair[0] as usize, pair[1] as usize);
                    for t in start..end {
                        let (lo, hi) = match mask {
                            Mask::Global => (start, end),
                            Mask::Causal => (start, t + 1),
                            Mask::Window { left, right } => {
                                (start.max(t.saturating_sub(left)), end.min(t + right + 1))
                            }
                            Mask::Local64 => (start.max(t.saturating_sub(64)), end.min(t + 65)),
                        };
                        for head in 0..h {
                            let kh = head / (h / hk);
                            let mut scores = Vec::with_capacity(hi - lo);
                            for key in lo..hi {
                                let score: f32 = (0..d)
                                    .map(|i| {
                                        qv[(t * h + head) * d + i] * kv[(key * hk + kh) * d + i]
                                    })
                                    .sum();
                                scores.push(score * scale);
                            }
                            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            for x in &mut scores {
                                *x = (*x - max).exp();
                            }
                            let sum: f32 = scores.iter().sum();
                            for i in 0..d {
                                let expected: f32 = scores
                                    .iter()
                                    .enumerate()
                                    .map(|(j, w)| w * vv[((lo + j) * hk + kh) * d + i] / sum)
                                    .sum();
                                let observed = actual[(t * h + head) * d + i];
                                let tolerance = if dtype == DType::F16 { 0.002 } else { 0.012 };
                                assert!(
                                    observed.is_finite() && (observed - expected).abs() < tolerance,
                                    "{dtype:?} {mask:?}: {observed} vs {expected}"
                                );
                            }
                        }
                    }
                }
                let other = if dtype == DType::F16 {
                    DType::BF16
                } else {
                    DType::F16
                };
                assert!(flash_attn_varlen(&q, &k.to_dtype(other)?, &v, &lengths, mask).is_err());
            }
        }
    }
    Ok(())
}

#[test]
fn packed_masks_match_uniform_attention() -> Result<()> {
    let dev = Device::new_cuda_with_stream(0)?;
    let offsets = [0u32, 1, 8, 73, 202];
    let lengths = Seqlens::new(&offsets, &dev)?;
    for dtype in [DType::F16, DType::BF16] {
        for (h, hk, d, mask) in [
            (3, 3, 64, Mask::Global),
            (5, 5, 64, Mask::Window { left: 7, right: 13 }),
            (4, 1, 128, Mask::Causal),
            (16, 16, 64, Mask::Global),
            (12, 12, 64, Mask::Global),
            (12, 12, 64, Mask::Local64),
            (32, 8, 128, Mask::Causal),
            (16, 8, 128, Mask::Causal),
            (6, 3, 128, Mask::Causal),
            (16, 8, 128, Mask::Global),
            (6, 3, 128, Mask::Global),
        ] {
            let total = 202;
            let q = Tensor::zeros((total, h, d), dtype, &dev)?;
            let k = Tensor::zeros((total, hk, d), dtype, &dev)?;
            let values: Vec<f32> = (0..total * hk * d)
                .map(|i| ((i / (hk * d)) % 17) as f32 / 16.)
                .collect();
            let v = Tensor::from_vec(values, (total, hk, d), &dev)?.to_dtype(dtype)?;
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
                        Mask::Window { left, right } => (
                            start.max(token.saturating_sub(left)),
                            end.min(token + right + 1),
                        ),
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
    }
    for offsets in [&[1u32, 2][..], &[0, 0], &[0, 2, 1], &[0, u32::MAX], &[0]] {
        assert!(Seqlens::new(offsets, &dev).is_err());
    }
    Ok(())
}
