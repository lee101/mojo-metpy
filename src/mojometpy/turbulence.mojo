"""Port of `metpy/calc/turbulence.py` (MetPy 1.7.1), in upstream source order.

Upstream reduces along `axis` with `np.mean(..., axis=axis)` after
`make_take`-shaped indexing; that indexing is a stride, so it is expressed here
as `(n, stride, outer)` - the axis length, its element stride and the number of
independent lines - and every line is reduced on its own.  The Python wrapper
computes those three numbers, which is the only place the port restates that
abstraction numerically.

`np.mean` uses pairwise summation; these are single accumulators, so the
summation order moves the last bits and `tests/test_parity.py` pins each case
above the measured difference.

A `W`-lane load reads `W` consecutive elements, which is only the reduction
axis when `stride == 1`, so that is the only case that vectorises.  The
`stride > 1` path is the same arithmetic one element at a time.
"""

from std.math import sqrt
from mojometpy.simd import F, W
from mojometpy.types import Ptr


def _line_mean(a: Ptr, n: Int, stride: Int, o: Int) -> Float64:
    var base = (o // stride) * n * stride + (o % stride)
    var acc = F(0.0)
    var total = 0.0
    var i = 0
    if stride == 1:
        while i + W <= n:
            acc += a.unsafe_load[width=W](base + i)
            i += W
    while i < n:
        total += a[unsafe_offset=base + i * stride]
        i += 1
    return (total + acc.reduce_add()) / Float64(n)


def _line_product_mean(a: Ptr, b: Ptr, n: Int, stride: Int, o: Int) -> Float64:
    var base = (o // stride) * n * stride + (o % stride)
    var acc = F(0.0)
    var total = 0.0
    var i = 0
    if stride == 1:
        while i + W <= n:
            acc += a.unsafe_load[width=W](base + i) * b.unsafe_load[width=W](base + i)
            i += W
    while i < n:
        total += a[unsafe_offset=base + i * stride] * b[unsafe_offset=base + i * stride]
        i += 1
    return (total + acc.reduce_add()) / Float64(n)


def _line_square_mean(a: Ptr, mean: Float64, debias: Int, n: Int, stride: Int, o: Int) -> Float64:
    var base = (o // stride) * n * stride + (o % stride)
    var acc = F(0.0)
    var total = 0.0
    var i = 0
    if stride == 1:
        while i + W <= n:
            var x = a.unsafe_load[width=W](base + i)
            if debias == 1:
                x = x - F(mean)
            acc += x * x
            i += W
    while i < n:
        var x = a[unsafe_offset=base + i * stride]
        if debias == 1:
            x -= mean
        total += x * x
        i += 1
    return (total + acc.reduce_add()) / Float64(n)


# ---- turbulence.py:38 ----


def _get_perturbation(a: Ptr, n: Int, stride: Int, o: Int) -> Float64:
    """Upstream's `ts - mean` along `axis`; only its mean is needed here."""
    return _line_mean(a, n, stride, o)


def tke(u: Ptr, v: Ptr, w: Ptr, perturbation: Int, n: Int, stride: Int, outer: Int, dst: Ptr):
    # `perturbation` is upstream's `not perturbation` branch: with it off each
    # component is debiased by its own line mean before squaring.
    var debias = 0
    if perturbation == 0:
        debias = 1
    for o in range(outer):
        var mean_u = 0.0
        var mean_v = 0.0
        var mean_w = 0.0
        if debias == 1:
            mean_u = _get_perturbation(u, n, stride, o)
            mean_v = _get_perturbation(v, n, stride, o)
            mean_w = _get_perturbation(w, n, stride, o)
        var u_cont = _line_square_mean(u, mean_u, debias, n, stride, o)
        var v_cont = _line_square_mean(v, mean_v, debias, n, stride, o)
        var w_cont = _line_square_mean(w, mean_w, debias, n, stride, o)
        dst[unsafe_offset=o] = 0.5 * (u_cont + v_cont + w_cont)


# ---- turbulence.py:57 ----


def _kinematic_flux(
    vel: Ptr, b: Ptr, perturbation: Int, n: Int, stride: Int, o: Int
) -> Float64:
    var kf = _line_product_mean(vel, b, n, stride, o)
    if perturbation == 0:
        kf -= _line_mean(vel, n, stride, o) * _line_mean(b, n, stride, o)
    return kf


def friction_velocity(
    u: Ptr, w: Ptr, v: Ptr, has_v: Int, perturbation: Int, n: Int, stride: Int, outer: Int, dst: Ptr
):
    # `has_v` carries upstream's `if v is not None`; `v` is then a real buffer
    # the kernel never reads.
    for o in range(outer):
        var uw = _kinematic_flux(u, w, perturbation, n, stride, o)
        var kf = uw**2
        if has_v == 1:
            var vw = _kinematic_flux(v, w, perturbation, n, stride, o)
            kf += vw**2
        # Upstream squares in place and then takes two square roots: "this is
        # faster than np.power(..., 0.25)".
        dst[unsafe_offset=o] = sqrt(sqrt(kf))
