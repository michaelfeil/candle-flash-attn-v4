# candle-flash-attn-v4

Native Rust/Candle inference bindings for [FlashAttention-4](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute).
The attention kernels are upstream Dao-AILab code. This repository packages their
ahead-of-time exports and provides a Candle interface. Python is used to build
the kernels, not in the inference process.

**Experimental, not yet a release.** The first implementation extracts a tested
TEI prototype. Work continues on a general configuration API and architecture
coverage. No crates.io release or universal performance/accuracy claim yet.

## Current native bundle

Hopper SM90, FP16/BF16, packed self- and cross-attention. Head counts and softmax scale
are runtime parameters; the current export families are:

| Q/KV relationship | Head dimension | Mask |
|---|---:|---|
| Any positive equal head counts (MHA) | 64 | global |
| Any positive equal head counts (MHA) | 64 | inclusive asymmetric sliding window |
| Q heads = 4 x KV heads (including 4/1 MQA) | 128 | causal |

`AttentionConfig` accepts a finite custom softmax scale; the default is
`1/sqrt(head_dim)`. `Mask::Window { left, right }` selects inclusive distances.
`Mask::Local64` remains shorthand for a 64/64 window. Windows must fit signed
int32 and are clamped to the longest sequence before launch to avoid index
overflow without changing the mask.
Zero and negative scales are supported by transforming Q before the native call.
These cases allocate a temporary Q tensor; positive scales use Q directly.

Sequence lengths are dynamic. Contiguous head dimensions and aligned row strides
are required. CPU execution, backward, arbitrary masks, and FP8
are not exposed by this initial bundle. Unsupported inputs return an
error. The wrapper never silently switches to another attention implementation.

Upstream has architecture paths for SM8x, SM90, SM10x/11x, and SM12x. That does
**not** mean this initial SM90 bundle runs on all of them. Ampere/Ada and RTX
Blackwell exports, plus datacenter Blackwell's different scheduler/ABI, are the
next packaging work. Non-H100 hardware has not been tested here. SM75 is not a
target of this upstream FA4 implementation.

## Build

Use Linux, CUDA, Python with CUDA-compatible PyTorch, and a
C++17 compiler. Install build dependencies in an isolated environment:

```sh
python -m pip install -r requirements-build.txt
python scripts/build_aot.py build/sm90 --compile-only --export-only
python scripts/build_aot.py build/sm90 --link-only \
  --runtime-dir /path/to/cutlass/cute_dsl/lib \
  --ffi-root /path/to/site-packages/tvm_ffi
export FA4_NATIVE_LIB_DIR="$PWD/build/sm90"
export LD_LIBRARY_PATH="$FA4_NATIVE_LIB_DIR:${LD_LIBRARY_PATH:-}"
cargo test --test attention
```

`--compile-only` uses fake tensors to export SM90 kernels without a build GPU.
It does not establish runtime correctness; the manifest records whether export
executed kernels. Omit it to export and execute on a Hopper GPU.

Choose the CuTe runtime compatible with the deployment CUDA version. The export
and link phases can run separately so the C++ compiler matches the deployment
image's libc/libstdc++. The bundle contains `libfa4bridge.so`,
`libcute_dsl_runtime.so`, and `libtvm_ffi.so`; deploy them together. The versioned native entry point prevents linking a bundle with an incompatible
parameter ABI. Re-export old bundles before building. The generated
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
// q: [tokens, 12, 64], k/v: [tokens, 12, 64], CUDA FP16 or BF16
let lengths = Seqlens::new(&[0, 127, 640], q.device())?;
let output = flash_attn_varlen(&q, &k, &v, &lengths, Mask::Local64)?;
```

`Seqlens` validates CPU boundaries before uploading them once; no GPU-to-CPU
offset read is needed per attention call. The wrapper retains Candle pointer
guards, respects the caller's stream, and checks runtime compute capability.

### Cross-attention

`flash_attn_varlen_cross(q, k, v, q_lengths, kv_lengths, config)` accepts separate
validated boundary objects with the same batch count. Q and KV token totals may
differ. Masks align the bottom-right corners: query `i` is centered at key
`i + kv_length - q_length`. This also applies to sliding windows. Fully masked
rows produce zeros. Empty sequences are still rejected.

## Initial validation

The standalone release-profile Cargo test build passed. On H100, global,
causal and local masks across packed lengths 1/7/65/129 matched an independent
uniform-attention reference on a nondefault CUDA stream. FP16 and BF16 also
passed a full nonuniform CPU attention reference, including GQA/MQA head mapping, new head counts, asymmetric windows, and a
nondefault softmax scale.
Strided views with nonzero offsets matched contiguous outputs exactly. Invalid boundaries and
token-count mismatches were rejected. Compute Sanitizer reported zero errors,
including after a fresh AOT export/link using this repository's build script.
Unequal Q/KV lengths passed independent per-sequence mask references in both
precisions, including fully masked causal rows, asymmetric and oversized windows,
and mismatched-batch rejection. The existing self-attention tests also passed
against the new native ABI. These tests do not replace model-level accuracy
qualification.

## Feature-completeness target

The goal is a general wrapper for the pinned upstream FA4 API, rather than a
collection of model presets. The current crate is not feature-complete.

| Area | Current state / remaining work |
|---|---|
| Dtypes | FP16/BF16 implemented; architecture-specific FP8 and scale tensors pending |
| Head geometry | Runtime head counts for the families above; configurable dimensions, value dimensions, and all upstream GQA ratios pending |
| Layouts | Packed self/cross-attention implemented; native dense paths pending |
| Masks | Global, causal, finite two-sided windows for the families above; remaining combinations and one-sided windows pending |
| GPU architectures | SM90 runtime tested; SM80/SM120 compile probes only; full architecture-aware exports and dispatch pending |
| Decode and scheduling | Paged KV, split-KV, scheduler metadata and associated workspace lifecycle pending |
| Advanced forward | LSE, softcap, sinks, auxiliary tensors, custom score/mask functions and block sparsity need export/API coverage |
| Training | Backward exports and Candle autograd integration pending; forward-only is not training support |
| Distribution | Pinned AOT sources and manifests implemented; reproducible per-architecture packages and release CI pending |
| Qualification | Layer/reference tests implemented; broader shape/dtype/device tests, model accuracy and performance gates pending |

Coverage must follow upstream's actual per-architecture support. Unsupported
combinations must fail clearly; a successful cross-compilation is not a GPU
correctness or performance result. Python-defined custom operations require
build-time specialization before they can be called from native Rust.

FA3 kernel rebasing belongs in the separate `candle-flash-attn-v3` repository.

## Attribution

Wrapper code: MIT OR Apache-2.0. Upstream FlashAttention: BSD-3-Clause (see
`LICENSE-UPSTREAM`). No upstream attention-kernel authorship is claimed here.

### Experimental DeBERTa-v2/v3 relative attention

Build the native bundle with `scripts/build_aot.py --deberta` and enable the
Rust `deberta` feature. `deberta_attn_varlen` accepts packed FP16/BF16 d64
self-attention on SM90, validated `Seqlens`, a `RelativeBuckets` lookup, and
precomputed content-to-position / position-to-content tables. It uses the same
FA4 tiled online-softmax kernel with a score hook; it never allocates an
attention matrix or pads a sequence to the batch maximum.

Q/K/V use `[total_tokens, heads, 64]`; the two relative tables use
`[heads, total_tokens, 2 * relative_span]`. Scale K and the relative tables as
documented on the Rust function. The caller must preserve the model's scaling
and projection rules. Workspace is linear in total tokens for a fixed relative
span; it is not zero, and callers must still enforce a total-token budget.

This operation is inference-only and does not implement arbitrary masks,
cross-attention, or original DeBERTa-v1 semantics. Unsupported inputs return
errors. There is no padded fallback after an execution error. Tiled softmax is
numerically close to eager attention, not bitwise identical; model/task quality
must be qualified separately.
