use candle::{DType, Device, Result, Tensor};
use candle_flash_attn_v4::{flash_attn_varlen_cross, AttentionConfig, Mask, Seqlens};

#[test]
fn unequal_lengths_match_bottom_right_reference() -> Result<()> {
    let dev = Device::new_cuda_with_stream(0)?;
    let q_offsets = [0u32, 1, 8, 137];
    let k_offsets = [0u32, 7, 10, 75];
    let q_lengths = Seqlens::new(&q_offsets, &dev)?;
    let k_lengths = Seqlens::new(&k_offsets, &dev)?;
    for dtype in [DType::F16, DType::BF16] {
        for (h, hk, d, mask) in [
            (3, 3, 64, Mask::Global),
            (5, 5, 64, Mask::Window { left: 2, right: 1 }),
            (
                3,
                3,
                64,
                Mask::Window {
                    left: i32::MAX as usize,
                    right: i32::MAX as usize,
                },
            ),
            (4, 1, 128, Mask::Causal),
        ] {
            let q = Tensor::zeros((137, h, d), dtype, &dev)?;
            let k = Tensor::zeros((75, hk, d), dtype, &dev)?;
            let values: Vec<f32> = (0..75 * hk * d)
                .map(|i| ((i / (hk * d)) % 13 + (i / d) % hk) as f32 / 16.)
                .collect();
            let v = Tensor::from_vec(values, (75, hk, d), &dev)?.to_dtype(dtype)?;
            let output = flash_attn_varlen_cross(
                &q,
                &k,
                &v,
                &q_lengths,
                &k_lengths,
                AttentionConfig {
                    mask,
                    softmax_scale: None,
                },
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            for b in 0..3 {
                let qs = q_offsets[b] as usize;
                let ks = k_offsets[b] as usize;
                let qn = (q_offsets[b + 1] - q_offsets[b]) as usize;
                let kn = (k_offsets[b + 1] - k_offsets[b]) as usize;
                for qi in 0..qn {
                    let center = qi as i64 + kn as i64 - qn as i64;
                    let visible: Vec<usize> = (0..kn)
                        .filter(|&ki| match mask {
                            Mask::Global => true,
                            Mask::Causal => ki as i64 <= center,
                            Mask::Window { left, right } => {
                                ki as i64 >= center - left as i64
                                    && ki as i64 <= center + right as i64
                            }
                            Mask::Local64 => unreachable!(),
                        })
                        .collect();
                    for head in 0..h {
                        let kh = head / (h / hk);
                        let expected = if visible.is_empty() {
                            0.
                        } else {
                            visible
                                .iter()
                                .map(|&ki| (((ks + ki) % 13) + kh) as f32 / 16.)
                                .sum::<f32>()
                                / visible.len() as f32
                        };
                        for &observed in
                            &output[((qs + qi) * h + head) * d..((qs + qi) * h + head + 1) * d]
                        {
                            assert!(
                                observed.is_finite() && (observed - expected).abs() < 0.004,
                                "{dtype:?} {mask:?} batch {b} query {qi}: {observed} vs {expected}"
                            );
                        }
                    }
                }
            }
            let wrong_batches = Seqlens::new(&[0, 75], &dev)?;
            assert!(flash_attn_varlen_cross(
                &q,
                &k,
                &v,
                &q_lengths,
                &wrong_batches,
                AttentionConfig {
                    mask,
                    softmax_scale: None
                }
            )
            .is_err());
        }
    }
    Ok(())
}
