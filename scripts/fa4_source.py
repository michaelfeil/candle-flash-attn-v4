"""Stage the pinned FA4 sources with FA2-compatible SM90 softmax arithmetic.

Only the private build copy is modified. The installed Python package remains
unchanged. Re-evaluate these compatibility choices when updating upstream.
"""
import hashlib
import importlib.util
import pathlib
import shutil
import sys
import tempfile

_ORIGINAL = "        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))"
_REPLACEMENT = """        # Match FA2's four-lane denominator reduction: offsets 2, then 1.
        # FP32 addition is not associative. Reversing the shuffles can change
        # FP16/BF16 rounding, amplified by subsequent encoder layers.
        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            total = row_sum[r]
            total = total + cute.arch.shuffle_sync_bfly(total, offset=2)
            total = total + cute.arch.shuffle_sync_bfly(total, offset=1)
            row_sum[r] = total"""

_TILE_ORIGINAL = """    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)"""
_TILE_REPLACEMENT = """    elif head_dim <= 128:
        # Match FA2's online-softmax block boundaries for causal d128.
        # A different key tile changes low-precision probability rounding.
        tile_n = 64 if head_dim == 128 and is_causal else 128
        return FwdConfig(128, tile_n, True, True)"""


def stage_fa4_sources():
    if any(name == "flash_attn" or name.startswith("flash_attn.") for name in sys.modules):
        raise RuntimeError("Stage the FA4 backport before importing flash_attn")
    spec = importlib.util.find_spec("flash_attn")
    locations = [] if spec is None else list(spec.submodule_search_locations or [])
    candidates = [pathlib.Path(p) for p in locations if (pathlib.Path(p) / "cute/softmax.py").is_file()]
    if len(candidates) != 1:
        raise RuntimeError("Install one pinned FA4 build package before exporting")
    original = candidates[0]
    text = (original / "cute/softmax.py").read_text()
    if text.count(_ORIGINAL) != 1:
        raise RuntimeError("FA4 softmax changed: review the denominator backport before exporting")
    patched = text.replace(_ORIGINAL, _REPLACEMENT)
    interface = (original / "cute/interface.py").read_text()
    if interface.count(_TILE_ORIGINAL) != 1:
        raise RuntimeError("FA4 SM90 tile selection changed: review the causal d128 backport")
    patched_interface = interface.replace(_TILE_ORIGINAL, _TILE_REPLACEMENT)
    stage = tempfile.TemporaryDirectory(prefix="fa4-softmax-")
    target = pathlib.Path(stage.name) / "flash_attn"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "cute/softmax.py").write_text(patched)
    (target / "cute/interface.py").write_text(patched_interface)
    sys.path.insert(0, stage.name)
    return stage, {
        "name": "fa2-compatible-four-lane-softmax-reduction",
        "original_softmax_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "patched_softmax_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "sm90_causal_d128_tile_n": 64,
        "original_interface_sha256": hashlib.sha256(interface.encode()).hexdigest(),
        "patched_interface_sha256": hashlib.sha256(patched_interface.encode()).hexdigest(),
    }
