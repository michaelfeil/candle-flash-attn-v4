"""Stage the pinned FA4 sources with the FA2-compatible denominator reduction.

Only the private build copy is modified. The installed Python package remains
unchanged. Remove this backport once the pinned upstream implements this order.
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
    stage = tempfile.TemporaryDirectory(prefix="fa4-softmax-")
    target = pathlib.Path(stage.name) / "flash_attn"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "cute/softmax.py").write_text(patched)
    sys.path.insert(0, stage.name)
    return stage, {
        "name": "fa2-compatible-four-lane-softmax-reduction",
        "original_softmax_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "patched_softmax_sha256": hashlib.sha256(patched.encode()).hexdigest(),
    }
