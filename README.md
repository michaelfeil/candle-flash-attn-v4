# candle-flash-attn-v4

Native Rust/Candle inference bindings for [FlashAttention-4](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute).
The attention kernels are upstream Dao-AILab code. This repository packages their
ahead-of-time exports and provides a Candle interface. Python is used to build
the kernels, not in the inference process.

**Experimental, not yet a release.** The first implementation extracts a tested
TEI prototype. Work continues on a general configuration API and architecture
coverage. No crates.io release or universal performance/accuracy claim yet.

## Current native bundle

Hopper SM90, FP16, packed self-attention, default `1/sqrt(head_dim)` scale:

| Q heads | KV heads | Head dimension | Mask |
|---:|---:|---:|---|
| 12 or 16 | same | 64 | global |
| 12 | 12 | 64 | local, inclusive +/-64 |
| 32 | 8 | 128 | causal |

Sequence lengths are dynamic. Contiguous head dimensions and aligned row strides
are required. CPU execution, backward, cross-attention, arbitrary masks, FP8,
and BF16 are not exposed by this initial bundle. Unsupported inputs return an
error. The wrapper never silently switches to another attention implementation.

Upstream has architecture paths for SM8x, SM90, SM10x/11x, and SM12x. That does
**not** mean this initial SM90 bundle runs on all of them. Ampere/Ada and RTX
Blackwell exports, plus datacenter Blackwell's different scheduler/ABI, are the
next packaging work. Non-H100 hardware has not been tested here. SM75 is not a
target of this upstream FA4 implementation.

## Build

Use Linux, a Hopper build GPU, CUDA, Python with CUDA-compatible PyTorch, and a
C++17 compiler. Install build dependencies in an isolated environment:

```sh
python -m pip install -r requirements-build.txt
python scripts/build_aot.py build/sm90 --export-only
python scripts/build_aot.py build/sm90 --link-only \
  --runtime-dir /path/to/cutlass/cute_dsl/lib \
  --ffi-root /path/to/site-packages/tvm_ffi
export FA4_NATIVE_LIB_DIR="$PWD/build/sm90"
export LD_LIBRARY_PATH="$FA4_NATIVE_LIB_DIR:${LD_LIBRARY_PATH:-}"
cargo test --test attention
```

Choose the CuTe runtime compatible with the deployment CUDA version. The export
and link phases can run separately so the C++ compiler matches the deployment
image's libc/libstdc++. The bundle contains `libfa4bridge.so`,
`libcute_dsl_runtime.so`, and `libtvm_ffi.so`; deploy them together. The generated
manifest records source/object/library hashes and tool versions. Native
dependencies retain their own licenses; preserve these when redistributing.

Upstream is pinned to `e9cf2c1651d2303191eb40a739a3c135fda00999`, CUTLASS DSL
4.7.1, and TVM FFI 0.1.14.post1. Candle is temporarily pinned to the revision
used by the native TEI prototype while standalone compatibility is validated.

## Compile-only architecture probes

```sh
python scripts/probe_arch.py 80 build/probe-sm80
python scripts/probe_arch.py 120 build/probe-sm120
```

Both exported a global d64 FP16 kernel successfully during development. These
use fake tensors and do not execute kernels. The resulting objects are probes,
not complete native bundles, and are not evidence of numerical correctness or
performance on those architectures.

## Rust interface

```rust,ignore
use candle_flash_attn_v4::{flash_attn_varlen, Mask, Seqlens};
// q: [tokens, 12, 64], k/v: [tokens, 12, 64], CUDA FP16
let lengths = Seqlens::new(&[0, 127, 640], q.device())?;
let output = flash_attn_varlen(&q, &k, &v, &lengths, Mask::Local64)?;
```

`Seqlens` validates CPU boundaries before uploading them once; no GPU-to-CPU
offset read is needed per attention call. The wrapper retains Candle pointer
guards, respects the caller's stream, and checks runtime compute capability.

## Initial validation

The standalone release-profile Cargo test build passed. On H100, global,
causal and local masks across packed lengths 1/7/65/129 matched an independent
uniform-attention reference on a nondefault CUDA stream. Invalid boundaries and
token-count mismatches were rejected. Compute Sanitizer reported zero errors,
including after a fresh AOT export/link using this repository's build script.
This focused test does not replace model-level accuracy qualification.

## Development priorities

1. Standalone build, mask/layout/stream tests, and reproducible native bundles.
2. Configuration-based exports instead of model-specific presets, FP16/BF16.
3. Architecture-specific build and dispatch; distinguish compilation from GPU validation.
4. Model-level numerical and performance checks before enabling FA4 in TEI.

FA3 kernel rebasing belongs in the separate `candle-flash-attn-v3` repository.

## Attribution

Wrapper code: MIT OR Apache-2.0. Upstream FlashAttention: BSD-3-Clause (see
`LICENSE-UPSTREAM`). No upstream attention-kernel authorship is claimed here.
