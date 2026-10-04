"""Replay every parity case with hostile (non-contiguous / wrong-dtype) arrays
and trace which pointers reach the FFI un-normalised."""
import sys
import numpy as np

sys.path.insert(0, "probe-tmp")
sys.path.insert(0, "tests")

import mojometpy._ffi as F
import test_parity as TP

current = [None]
bad = []


def _is_clean(p):
    if isinstance(p, np.ndarray):
        return p.dtype == np.float64 and p.flags["C_CONTIGUOUS"]
    return not isinstance(p, np.ndarray)


def traced(p):
    if not _is_clean(p):
        bad.append((current[0], type(p).__name__, getattr(p, "dtype", None),
                    getattr(p, "shape", None), bool(np.asarray(p).flags["C_CONTIGUOUS"])
                    if isinstance(p, np.ndarray) else None))
    return _orig_arg(p)


_orig_arg = F._arg
F._arg = traced
# kinematics/thermo call _lib.lib().sym(...) directly, bypassing _arg; patch _arg
# only catches the helpers.  Patch the CDLL too via a recording argtypes hook.
import mojometpy._lib as L

_real_lib = L.lib()
seen_direct = []


class Rec:
    def __init__(self, fn, name):
        self._fn = fn
        self._name = name

    def __call__(self, *a):
        arrs = [x for x in a if isinstance(x, (int, np.integer))]
        for x in a:
            if isinstance(x, np.ndarray):
                arrs.append(x)
        for x in a:
            if isinstance(x, np.ndarray) and not (
                x.dtype == np.float64 and x.flags["C_CONTIGUOUS"]
            ):
                bad.append((current[0], "DIRECT " + self._name, x.dtype, x.shape,
                            x.flags["C_CONTIGUOUS"]))
        return self._fn(*a)

    def __getattr__(self, k):
        return Rec(getattr(self._fn, k), k)


_real_lib.__class__  # noqa
_rec = Rec(_real_lib, "<root>")
L._lib = _rec
import mojometpy.kinematics as K
import mojometpy.thermo as TH
import mojometpy.basic as B
K._lib = _rec
TH._lib = _rec
B._lib = _rec


def poison(d):
    """Make every ndarray in the fixture non-contiguous float64, and every
    float64 array an int64 one, so a missing normalisation shows up."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            w = np.zeros(v.shape, dtype=np.float64) + v
            out[k] = np.ascontiguousarray(w)[:, ::1].copy(order="C")
            # step 2 -> strided, same shape
            big = np.zeros(tuple(2 if s else s for s in w.shape), dtype=np.float64)
            sl = tuple(slice(None, None, 2) for _ in w.shape)
            big[sl] = w
            out[k] = big[sl]
        else:
            out[k] = v
    return out


poison_mode = sys.argv[1] if len(sys.argv) > 1 else "strided"


def apply_poison(v):
    if not isinstance(v, np.ndarray):
        return v
    if poison_mode == "strided":
        big = np.zeros(tuple(2 * s for s in v.shape) or (2,), dtype=np.float64)
        sl = tuple(slice(None, None, 2) for _ in v.shape) or (slice(None, None, 2),)
        big[sl] = v
        return big[sl]
    if poison_mode == "int":
        return v.astype(np.int64)
    if poison_mode == "f32":
        return v.astype(np.float32)
    return v


# monkeypatch the fixture: wrap every array value on access
class PoisonDict(dict):
    def __getitem__(self, k):
        return apply_poison(dict.__getitem__(self, k))


d0 = TP.d.__wrapped__()
d = PoisonDict(d0)

base_bad = len(bad)
failures = []
for name, fn, rtol, atol in TP.CASES:
    current[0] = name
    n0 = len(bad)
    try:
        ours, theirs = fn(d)
    except Exception as e:
        failures.append((name, f"{type(e).__name__}: {str(e)[:70]}"))
        continue
    if len(bad) > n0:
        entries = bad[n0:]
        # dedupe
        uniq = sorted(set(entries))
        failures.append((name, "UNNORMALISED PTR -> " + "; ".join(
            f"{e[1]} dtype={e[2]} shape={e[3]} contig={e[4]}" for e in uniq[:3])))

print(f"poison mode: {poison_mode}")
print(f"cases: {len(TP.CASES)}  with problems: {len(failures)}")
for n, why in failures:
    print(f"  {n}: {why}")
