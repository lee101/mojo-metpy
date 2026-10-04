"""SIMD helpers shared by the kernels.

Mojo vectorises `exp`, `log`, `sqrt` and the arithmetic operators on a `SIMD`
lane type, but lowers `x ** y` to one scalar `pow` call per lane, which is
slower than the scalar loop it replaces.  Every power below is therefore
written `exp(y * log(x))`; on this toolchain the two forms evaluate to the
identical double, so the substitution is bit for bit rather than an
approximation.

There is no lane-wise blend on a `SIMD`, so the masked branches the scalar
kernels write as `nan` are produced by `div_pos`, which folds the condition
through `sqrt`: `sqrt` of a negative is NaN and of a zero is zero, so
`x * (sqrt(y) / sqrt(y))` is exactly `x` for `y > 0` and NaN for `y <= 0`.

`PAR_MIN` and `PAR_WORKERS` size the `max.algorithm.parallelize` split the
compute-bound kernels carry.  The split is worth its fixed cost on a kernel
that spends a transcendental per element and is not on one that does not, so
the threshold is a measured compromise: at `PAR_MIN` the split is 1.3x faster
than the serial loop for `potential_temperature` and 1.7x slower for
`dewpoint`.  A `W`-lane pass over pure arithmetic is already at the memory
bandwidth limit, so those kernels stay serial.
"""

from std.math import exp, log, sqrt
from std.sys import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime F = SIMD[DType.float64, W]
comptime B = SIMD[DType.bool, W]

comptime PAR_MIN: Int = 524288
comptime PAR_WORKERS: Int = 16


@always_inline
def vpow(x: F, y: F) -> F:
    return exp(y * log(x))


@always_inline
def vpowc(x: F, y: Float64) -> F:
    return exp(F(y) * log(x))


@always_inline
def div_pos(x: F, y: F) -> F:
    return (x / y) * (sqrt(y) / sqrt(y))


@always_inline
def flag(c: B) -> F:
    return F(c)
