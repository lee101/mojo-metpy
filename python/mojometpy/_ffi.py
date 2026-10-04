"""Plumbing shared by the public modules.

Two jobs: turn Python arguments into the flat C argument list `src/capi.mojo`
expects, and rebuild the upstream result shape.  Every function here is
agnostic about what the kernels compute - all meteorological logic lives in
`src/mojometpy/*.mojo`.
"""

from __future__ import annotations

import numpy as np

from . import _lib

# Upstream's `phase` keyword as the integer the kernels switch on.
PHASE = {"liquid": 0, "solid": 1, "auto": 2}


def phase_code(phase: str) -> int:
    try:
        return PHASE[phase]
    except KeyError:
        raise ValueError(
            f"{phase!r} is not a valid option for phase. "
            f"Valid options are {tuple(PHASE)}."
        ) from None


def mag(x):
    """Value of `x` as float64, unwrapping a pint `Quantity` if given."""
    if hasattr(x, "magnitude"):
        x = x.magnitude
    return np.asarray(x, dtype=np.float64)


def bcast(*arrays):
    """Broadcast the given arrays to a common shape, C-contiguous."""
    arrays = [mag(a) for a in arrays]
    shape = np.broadcast_shapes(*[a.shape for a in arrays])
    return [
        np.ascontiguousarray(np.broadcast_to(a, shape), dtype=np.float64)
        for a in arrays
    ], shape


def size(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def call(sym: str, *parts, shape) -> np.ndarray:
    """Call a kernel with a trailing `(n, out)` and reshape the result."""
    n = size(shape)
    out = np.empty(n, dtype=np.float64)
    args = [_arg(p) for p in parts]
    args.append(n)
    args.append(out.ctypes.data)
    getattr(_lib.lib(), sym)(*args)
    return out.reshape(shape)


def call_scalar(sym: str, *parts, n: int) -> float:
    """Call a kernel that reduces `n` elements into a single scalar.

    The kernel only ever writes element 0, so a one-element buffer is enough
    and the call does not allocate a full-length result it would throw away.
    """
    out = np.empty(1, dtype=np.float64)
    args = [_arg(p) for p in parts]
    args.append(int(n))
    args.append(out.ctypes.data)
    getattr(_lib.lib(), sym)(*args)
    return float(out[0])


def call3(sym: str, *parts, n: int) -> tuple[float, float, float]:
    """Call a kernel that reduces `n` elements into three scalars."""
    out = np.empty(3, dtype=np.float64)
    args = [_arg(p) for p in parts]
    args.append(int(n))
    args.append(out.ctypes.data)
    getattr(_lib.lib(), sym)(*args)
    return float(out[0]), float(out[1]), float(out[2])


def call_int(sym: str, *parts) -> int:
    """Call a kernel that returns a single integer."""
    return int(getattr(_lib.lib(), sym)(*[_arg(p) for p in parts]))


def call_axis(sym: str, *parts, shape, axis: int) -> np.ndarray:
    """Call a kernel that shortens `shape` by one along `axis`."""
    n, stride, outer = axis_geometry(shape, axis)
    axis = axis + len(shape) if axis < 0 else axis
    out_shape = shape[:axis] + (shape[axis] - 1,) + shape[axis + 1:]
    out = np.empty(outer * (n - 1), dtype=np.float64)
    args = [_arg(p) for p in parts]
    args += [int(n), int(stride), int(outer), out.ctypes.data]
    getattr(_lib.lib(), sym)(*args)
    return out.reshape(out_shape)



def call2(sym: str, *parts, shape):
    """Call a kernel that writes two trailing output buffers."""
    n = size(shape)
    a = np.empty(n, dtype=np.float64)
    b = np.empty(n, dtype=np.float64)
    args = [_arg(p) for p in parts]
    args.append(n)
    args.append(a.ctypes.data)
    args.append(b.ctypes.data)
    getattr(_lib.lib(), sym)(*args)
    return a.reshape(shape), b.reshape(shape)


def call_masked(sym: str, *parts, shape):
    """Call a kernel that writes a value buffer and a 0/1 mask buffer."""
    n = size(shape)
    values = np.empty(n, dtype=np.float64)
    mask = np.empty(n, dtype=np.float64)
    args = [_arg(p) for p in parts]
    args.append(n)
    args.append(values.ctypes.data)
    args.append(mask.ctypes.data)
    getattr(_lib.lib(), sym)(*args)
    return values.reshape(shape), mask.reshape(shape) != 0.0


def with_mask(values: np.ndarray, mask: np.ndarray):
    """Upstream only returns a masked array when something is actually masked."""
    if mask.any():
        return np.ma.MaskedArray(values, mask=mask)
    return values


def _arg(p):
    if isinstance(p, np.ndarray):
        return int(p.ctypes.data)
    if isinstance(p, (bool, np.bool_)):
        return int(p)
    if isinstance(p, (int, np.integer)):
        return int(p)
    if isinstance(p, (float, np.floating)):
        return float(p)
    raise TypeError(f"cannot pass {type(p)!r} across the FFI")


def axis_geometry(shape, axis: int) -> tuple[int, int, int]:
    """`(n, stride, outer)` for one axis of a C-contiguous array.

    Upstream indexes with `make_take(ndim, axis)`, which is exactly a stride;
    this is the one place the port restates that abstraction numerically.
    """
    if axis < 0:
        axis += len(shape)
    n = shape[axis]
    stride = size(shape[axis + 1:])
    outer = size(shape) // n
    return n, stride, outer
