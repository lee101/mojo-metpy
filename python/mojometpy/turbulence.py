"""Port of `metpy/calc/turbulence.py` (MetPy 1.7.1).

Both functions reduce along `axis`; the `(n, stride, outer)` geometry that says
so is computed here, because upstream expresses it as `make_take` indexing
inside `np.mean(..., axis=axis)`.
"""

from __future__ import annotations

import numpy as np

from ._ffi import axis_geometry, bcast, size


def _reduce_args(u, v, w, axis):
    arrays, shape = bcast(u, v, w)
    axis = axis + len(shape) if axis < 0 else axis
    n, stride, outer = axis_geometry(shape, axis)
    out_shape = shape[:axis] + shape[axis + 1:]
    return arrays, n, stride, outer, size(out_shape), out_shape


def _reduce_call(sym, arrays, n, stride, outer, total, out_shape, *flags):
    from . import _lib

    out = np.empty(total, dtype=np.float64)
    # attribute lookup, not `__getattr__`: the latter hands back an unconfigured
    # ctypes thunk with no argtypes, which segfaults on a 64-bit address
    getattr(_lib.lib(), sym)(
        *(a.ctypes.data for a in arrays), *flags, n, stride, outer, out.ctypes.data
    )
    return out.reshape(out_shape)


def tke(u, v, w, perturbation=False, axis=-1):
    """Upstream `turbulence.tke`: half the mean of the three squared components."""
    args, n, stride, outer, total, out_shape = _reduce_args(u, v, w, axis)
    return _reduce_call(
        "mm_tke", args, n, stride, outer, total, out_shape, 1 if perturbation else 0
    )


def friction_velocity(u, w, v=None, perturbation=False, axis=-1):
    """Upstream `turbulence.friction_velocity`; v is optional, as upstream."""
    if v is None:
        args, n, stride, outer, total, out_shape = _reduce_args(u, w, w, axis)
        v_arg = args[2]
        has_v = 0
    else:
        args, n, stride, outer, total, out_shape = _reduce_args(u, w, v, axis)
        v_arg = args[2]
        has_v = 1
    u_arg, w_arg, _ = args
    return _reduce_call(
        "mm_friction_velocity", (u_arg, w_arg, v_arg), n, stride, outer, total,
        out_shape, has_v, 1 if perturbation else 0
    )
