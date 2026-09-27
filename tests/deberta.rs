#![cfg(feature = "deberta")]
use candle::{DType, Device, Result, Tensor};
use candle_flash_attn_v4::{deberta_attn_varlen, RelativeBuckets, Seqlens};

#[test]
fn packed_relative_attention_matches_independent_reference() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let lengths = [1usize, 7, 33, 65];
    let mut cu = vec![0u32];
    for n in lengths {
        cu.push(cu.last().unwrap() + n as u32);
    }
    let total = *cu.last().unwrap() as usize;
    let heads = 4;
    let span = 16;
    for dtype in [DType::F16, DType::BF16] {
        let q = Tensor::zeros((total, heads, 64), dtype, &dev)?;
        let v: Vec<f32> = (0..total * heads * 64)
            .map(|i| ((i * 17 % 101) as f32 - 50.) / 64.)
            .collect();
        let values = Tensor::from_vec(v.clone(), (total, heads, 64), &dev)?.to_dtype(dtype)?;
        let table = |seed: usize| -> Vec<f32> {
            (0..heads * total * 2 * span)
                .map(|i| ((i * seed % 31) as f32 - 15.) / 32.)
                .collect()
        };
        let a = table(7);
        let b = table(13);
        let at = Tensor::from_vec(a.clone(), (heads, total, 2 * span), &dev)?.to_dtype(dtype)?;
        let bt = Tensor::from_vec(b.clone(), (heads, total, 2 * span), &dev)?.to_dtype(dtype)?;
        let seq = Seqlens::new(&cu, &dev)?;
        let buckets = RelativeBuckets::new(65, span, false, 512, &dev)?;
        let actual = deberta_attn_varlen(&q, &q, &values, &at, &bt, &seq, &buckets)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut start = 0;
        for n in lengths {
            for i in 0..n {
                for h in 0..heads {
                    let scores: Vec<f64> = (0..n)
                        .map(|j| {
                            // Separate content->position and transposed position->content
                            // lookup from the mathematical definition, including clipping.
                            let forward = (i as isize - j as isize + span as isize)
                                .clamp(0, (2 * span - 1) as isize)
                                as usize;
                            let reverse = (-(j as isize - i as isize) + span as isize)
                                .clamp(0, (2 * span - 1) as isize)
                                as usize;
                            ((a[(h * total + start + i) * 2 * span + forward]
                                + b[(h * total + start + j) * 2 * span + reverse])
                                as f64)
                                .exp()
                        })
                        .collect();
                    let sum: f64 = scores.iter().sum();
                    for d in 0..64 {
                        let expected: f64 = (0..n)
                            .map(|j| scores[j] / sum * v[((start + j) * heads + h) * 64 + d] as f64)
                            .sum();
                        let actual = actual[((start + i) * heads + h) * 64 + d] as f64;
                        assert!(actual.is_finite());
                        assert!(
                            (actual - expected).abs()
                                < if dtype == DType::F16 { 0.001 } else { 0.008 },
                            "{dtype:?} n={n} i={i} h={h} d={d}: {actual} vs {expected}"
                        );
                    }
                }
            }
            start += n;
        }
        let too_short = RelativeBuckets::new(64, span, false, 512, &dev)?;
        assert!(deberta_attn_varlen(&q, &q, &values, &at, &bt, &seq, &too_short).is_err());
        assert!(deberta_attn_varlen(
            &q,
            &q,
            &values,
            &at.narrow(2, 0, span)?,
            &bt,
            &seq,
            &buckets
        )
        .is_err());
    }
    Ok(())
}

// HF rounds c2p+p2c before adding QK. The alternate sequential grouping
// produces a nonuniform softmax here, while the reference yields exactly 1/2.
#[test]
fn relative_bias_rounds_before_adding_nonzero_content() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for dtype in [DType::F16, DType::BF16] {
        let mut qk = vec![0f32; 128];
        qk[0] = 16.;
        qk[64] = 16.;
        let qk = Tensor::from_vec(qk, (2, 1, 64), &dev)?.to_dtype(dtype)?;
        let values = Tensor::from_vec([vec![0f32; 64], vec![1f32; 64]].concat(), (2, 1, 64), &dev)?
            .to_dtype(dtype)?;
        let c2p = Tensor::from_vec(vec![-256f32; 8], (1, 2, 4), &dev)?.to_dtype(dtype)?;
        let delta = if dtype == DType::F16 { 0.0625 } else { 0.25 };
        let p2c = Tensor::from_vec([vec![-delta; 4], vec![delta; 4]].concat(), (1, 2, 4), &dev)?
            .to_dtype(dtype)?;
        let seq = Seqlens::new(&[0, 2], &dev)?;
        let buckets = RelativeBuckets::new(2, 2, false, 512, &dev)?;
        let output = deberta_attn_varlen(&qk, &qk, &values, &c2p, &p2c, &seq, &buckets)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(
            output.iter().all(|v| (v - 0.5).abs() < 0.001),
            "{dtype:?}: {output:?}"
        );
    }
    Ok(())
}

#[test]
fn runtime_sequences_can_exceed_the_export_example() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let lens = [513usize, 1, 2049];
    let mut cu = vec![0u32];
    let mut values = Vec::new();
    let mut means = Vec::new();
    for (seq, &n) in lens.iter().enumerate() {
        cu.push(cu.last().unwrap() + n as u32);
        let data: Vec<f32> = (0..n)
            .map(|i| seq as f32 - 1. + (i % 4) as f32 / 4.)
            .collect();
        means.push(data.iter().sum::<f32>() / n as f32);
        for value in data {
            values.extend([value; 64]);
        }
    }
    let total = *cu.last().unwrap() as usize;
    for dtype in [DType::F16, DType::BF16] {
        let qk = Tensor::zeros((total, 1, 64), dtype, &dev)?;
        let v = Tensor::from_vec(values.clone(), (total, 1, 64), &dev)?.to_dtype(dtype)?;
        let rel = Tensor::zeros((1, total, 4), dtype, &dev)?;
        let seq = Seqlens::new(&cu, &dev)?;
        let buckets = RelativeBuckets::new(2049, 2, false, 512, &dev)?;
        let output = deberta_attn_varlen(&qk, &qk, &v, &rel, &rel, &seq, &buckets)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // Uniform per-sequence attention has an analytic result. Check every
        // token, including tiles well beyond the 129-token export example.
        for (i, &mean) in means.iter().enumerate() {
            let tolerance = if dtype == DType::F16 { 0.002 } else { 0.016 };
            assert!(
                output[cu[i] as usize * 64..cu[i + 1] as usize * 64]
                    .iter()
                    .all(|v| v.is_finite() && (v - mean).abs() < tolerance),
                "{dtype:?} sequence {i} did not match its own mean {mean}"
            );
        }
    }
    Ok(())
}
