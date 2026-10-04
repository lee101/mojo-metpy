"""Port of `metpy/calc/tools.py` and the interpolation and smoothing kernels of
`metpy/calc/basic.py` (MetPy 1.7.1), in upstream source order.

`first_derivative` / `second_derivative` are upstream's three-region (left,
center, right) centered-difference stencils.  Upstream slices an N-d array
along one axis; the same thing is expressed here as `(n, stride, outer)`, the
axis length, its element stride and the number of independent lines, so any
rank works without a special case.  The Python wrapper computes those three
numbers, which is the only place the port diverges structurally.

`delta` carries upstream's `_broadcast_to_axis` contract, so it is always a
length-`n-1` vector shared by every line and never shares `f`'s geometry.

A `SIMD` load reads `W` consecutive elements.  For the last axis
(`stride == 1`) consecutive elements are consecutive axis positions, so the
lanes run along the axis.  For every other stride the lines are `n * stride`
apart and no contiguous load walks them, so each line is reduced on its own.
The three point ends follow whichever nesting is in use.

`normalize_window` is upstream's `window / np.sum(window)`.

`smooth_window` is upstream's two-pass weighted sum, with the weights already
normalised by the wrapper (`normalize_weights`, upstream's default).  A `W`-lane
load gathers `W` taps of one window element, one window row per lane, so the
vector path is taken only when the window height is the lane width; the scalar
loop below is upstream's (ky, kx) order and covers every other shape.
"""

from std.math import abs, atan2, cos, sin, sqrt
from mojometpy.constants import geod_sphere_radius
from mojometpy.simd import F, W, flag
from mojometpy.types import Ptr

comptime DEG2RAD: Float64 = 3.141592653589793 / 180.0
comptime HALF_PI: Float64 = 1.5707963267948966


# ---- interpolation (metpy/interpolate/one_dimension.py:52) ----

# Upstream sorts `xp` and `x` with `argsort` and gathers through the resulting
# index arrays, then broadcasts the per-axis indices across the other
# dimensions; here `xp` is sorted by the wrapper and the same broadcast is the
# `(n, stride, outer)` line geometry every other kernel here uses.  The
# `searchsorted` inside is a binary search whose result is data dependent, so
# the pass is scalar: the indices cannot be gathered by a `W`-lane load.
def interpolate_1d(
    x: Ptr,
    xp: Ptr,
    values: Ptr,
    m: Int,
    n: Int,
    stride: Int,
    outer: Int,
    fill_value: Float64,
    dst: Ptr,
):
    for o in range(outer):
        var xb = (o // stride) * m * stride + (o % stride)
        var pb = (o // stride) * n * stride + (o % stride)
        for j in range(m):
            var xv = x[unsafe_offset=j]
            # `np.searchsorted` default, side='left'
            var lo = 0
            var hi = n
            while lo < hi:
                var mid = (lo + hi) // 2
                if xp[unsafe_offset=pb + mid * stride] < xv:
                    lo = mid + 1
                else:
                    hi = mid
            var minv = lo
            # Points off either end are pulled back onto the end interval, then
            # filled: upstream's `minv2` and its two out-of-bounds rules.
            var minv2 = minv
            if minv == n:
                minv2 = n - 1
            elif minv == 0:
                minv2 = 1
            var below = minv2 - 1
            var above = minv2
            var out = values[unsafe_offset=pb + below * stride] + (
                values[unsafe_offset=pb + above * stride]
                - values[unsafe_offset=pb + below * stride]
            ) * (
                (xv - xp[unsafe_offset=pb + below * stride])
                / (xp[unsafe_offset=pb + above * stride] - xp[unsafe_offset=pb + below * stride])
            )
            if minv == n or xv < xp[unsafe_offset=pb + below * stride]:
                out = fill_value
            dst[unsafe_offset=xb + j * stride] = out


# ---- tools.py:52 ----


def resample_nn_1d_nearest(a: Ptr, center: Float64, n: Int) -> Int:
    """Upstream's `(np.abs(a - center)).argmin()` for one center."""
    var best = 0
    var best_d = abs(a[unsafe_offset=0] - center)
    for i in range(1, n):
        var d = abs(a[unsafe_offset=i] - center)
        if d < best_d:
            best_d = d
            best = i
    return best


# ---- tools.py:78 ----


@always_inline
def _sign(x: Float64) -> Float64:
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    else:
        return 0.0


def nearest_intersection_idx(a: Ptr, b: Ptr, n: Int, dst: Ptr) -> Int:
    """Indices where `np.diff(np.sign(a - b))` is nonzero."""
    var found = 0
    for i in range(1, n):
        # `np.sign` maps 0 to 0, so a difference passing exactly through zero is
        # two changes, exactly as `np.diff(np.sign(...))` counts it
        if _sign(a[unsafe_offset=i - 1] - b[unsafe_offset=i - 1]) != _sign(
            a[unsafe_offset=i] - b[unsafe_offset=i]
        ):
            dst[unsafe_offset=found] = Float64(i - 1)
            found += 1
    return found


# ---- tools.py:789 ----

# The spherical geodesic inverse, Vincenty's closed form for a sphere, which is
# what `pyproj.Geod(ellps='sphere').inv` evaluates.  `az` is the forward
# azimuth in radians; upstream flips the sign of the distance wherever the
# azimuth points the other way along the axis.
@always_inline
def _geod_inv(lon1: F, lat1: F, lon2: F, lat2: F) -> Tuple[F, F, F]:
    var dlon = (lon2 - lon1) * F(DEG2RAD)
    var rlat1 = lat1 * F(DEG2RAD)
    var rlat2 = lat2 * F(DEG2RAD)
    var slat1 = sin(rlat1)
    var clat1 = cos(rlat1)
    var slat2 = sin(rlat2)
    var clat2 = cos(rlat2)
    var coslam = cos(dlon)
    var costheta = slat1 * slat2 + clat1 * clat2 * coslam
    var sintheta = sqrt(F(1.0) - costheta * costheta)
    var az12 = atan2(sin(dlon) * clat2, clat1 * slat2 - slat1 * clat2 * coslam)
    var az21 = atan2(-sin(dlon) * clat1, clat2 * slat1 - slat2 * clat1 * coslam)
    var s12 = atan2(sintheta, costheta)
    return (az12, az21, s12 * F(geod_sphere_radius))


# Upstream's `dy[(forward_az < -90.) | (forward_az > 90.)] *= -1`.
@always_inline
def _dy_sign(az: F) -> F:
    return F(1.0) - F(2.0) * flag(az.lt(-HALF_PI) | az.gt(HALF_PI))


# Upstream's `dx[(forward_az < 0.) | (forward_az > 180.)] *= -1`; the azimuth
# out of `atan2` is in (-180, 180], so only the negative half can trigger it.
@always_inline
def _dx_sign(az: F) -> F:
    return F(1.0) - F(2.0) * flag(az.lt(0.0))


def lat_lon_grid_deltas_dy(
    longitude: Ptr, latitude: Ptr, n: Int, stride: Int, outer: Int, dst: Ptr
):
    # A `W`-lane load reads `W` consecutive elements, so the lanes only run
    # along the axis when it is the last one.
    var o = 0
    while o < outer:
        var base = (o // stride) * n * stride + (o % stride)
        var i = 0
        if stride == 1:
            while i + W <= n - 1:
                var az, az21, dist = _geod_inv(
                    longitude.unsafe_load[width=W](base + i),
                    latitude.unsafe_load[width=W](base + i),
                    longitude.unsafe_load[width=W](base + i + 1),
                    latitude.unsafe_load[width=W](base + i + 1),
                )
                dst.unsafe_store(o * (n - 1) + i, dist * _dy_sign(az))
                i += W
        while i < n - 1:
            var az, az21, dist = _geod_inv(
                F(longitude[unsafe_offset=base + i * stride]),
                F(latitude[unsafe_offset=base + i * stride]),
                F(longitude[unsafe_offset=base + (i + 1) * stride]),
                F(latitude[unsafe_offset=base + (i + 1) * stride]),
            )
            dst[unsafe_offset=o * (n - 1) + i] = dist[0] * _dy_sign(az)[0]
            i += 1
        o += 1


def lat_lon_grid_deltas_dx(
    longitude: Ptr, latitude: Ptr, n: Int, stride: Int, outer: Int, dst: Ptr
):
    var o = 0
    while o < outer:
        var base = (o // stride) * n * stride + (o % stride)
        var i = 0
        if stride == 1:
            while i + W <= n - 1:
                var az, az21, dist = _geod_inv(
                    longitude.unsafe_load[width=W](base + i),
                    latitude.unsafe_load[width=W](base + i),
                    longitude.unsafe_load[width=W](base + i + 1),
                    latitude.unsafe_load[width=W](base + i + 1),
                )
                dst.unsafe_store(o * (n - 1) + i, dist * _dx_sign(az))
                i += W
        while i < n - 1:
            var az, az21, dist = _geod_inv(
                F(longitude[unsafe_offset=base + i * stride]),
                F(latitude[unsafe_offset=base + i * stride]),
                F(longitude[unsafe_offset=base + (i + 1) * stride]),
                F(latitude[unsafe_offset=base + (i + 1) * stride]),
            )
            dst[unsafe_offset=o * (n - 1) + i] = dist[0] * _dx_sign(az)[0]
            i += 1
        o += 1


# ---- derivative stencils (tools.py:1197) ----



def first_derivative(f: Ptr, delta: Ptr, n: Int, stride: Int, outer: Int, dst: Ptr):
    # `delta` is a length-`n-1` vector shared by every line, so a `W`-lane load
    # of `f` only gathers `W` *adjacent* elements, which are `W` adjacent lines
    # only when `stride == 1`.  For every other stride the lines are `n*stride`
    # apart and there is no contiguous load that walks them, so each line is
    # reduced on its own.
    var o = 0
    if stride == 1:
        while o < outer:
            _fd_line_v(f, delta, n, o, dst)
            o += 1
    else:
        while o < outer:
            _fd_line_s(f, delta, n, stride, o, dst)
            o += 1


def _fd_line_s(
    f: Ptr, delta: Ptr, n: Int, stride: Int, o: Int, dst: Ptr
):
    var base = (o // stride) * n * stride + (o % stride)
    var d0 = delta[unsafe_offset=(0)]
    var d1 = delta[unsafe_offset=(1)]
    var cd = d0 + d1
    var bd = cd + d0
    dst[unsafe_offset=base] = (
        -bd / (cd * d0) * f[unsafe_offset=base]
        + cd / (d0 * d1) * f[unsafe_offset=base + stride]
        - d0 / (cd * d1) * f[unsafe_offset=base + 2 * stride]
    )
    for i in range(1, n - 1):
        d0 = delta[unsafe_offset=(i - 1)]
        d1 = delta[unsafe_offset=(i)]
        cd = d0 + d1
        var dd = d1 - d0
        dst[unsafe_offset=base + i * stride] = (
            -d1 / (cd * d0) * f[unsafe_offset=base + (i - 1) * stride]
            + dd / (d0 * d1) * f[unsafe_offset=base + i * stride]
            + d0 / (cd * d1) * f[unsafe_offset=base + (i + 1) * stride]
        )
    d0 = delta[unsafe_offset=(n - 3)]
    d1 = delta[unsafe_offset=(n - 2)]
    cd = d0 + d1
    bd = cd + d1
    dst[unsafe_offset=base + (n - 1) * stride] = (
        d1 / (cd * d0) * f[unsafe_offset=base + (n - 3) * stride]
        - cd / (d0 * d1) * f[unsafe_offset=base + (n - 2) * stride]
        + bd / (cd * d1) * f[unsafe_offset=base + (n - 1) * stride]
    )


def _fd_line_v(f: Ptr, delta: Ptr, n: Int, o: Int, dst: Ptr):
    var base = o * n
    var d0 = delta[unsafe_offset=(0)]
    var d1 = delta[unsafe_offset=(1)]
    var cd = d0 + d1
    var bd = cd + d0
    dst[unsafe_offset=base] = (
        -bd / (cd * d0) * f[unsafe_offset=base]
        + cd / (d0 * d1) * f[unsafe_offset=base + 1]
        - d0 / (cd * d1) * f[unsafe_offset=base + 2]
    )
    var i = 1
    while i + W <= n - 1:
        var d0v = delta.unsafe_load[width=W]((i - 1))
        var d1v = delta.unsafe_load[width=W]((i))
        var cv = d0v + d1v
        var dd = d1v - d0v
        dst.unsafe_store(
            base + i,
            -d1v / (cv * d0v) * f.unsafe_load[width=W](base + i - 1)
            + dd / (d0v * d1v) * f.unsafe_load[width=W](base + i)
            + d0v / (cv * d1v) * f.unsafe_load[width=W](base + i + 1),
        )
        i += W
    while i < n - 1:
        d0 = delta[unsafe_offset=(i - 1)]
        d1 = delta[unsafe_offset=(i)]
        cd = d0 + d1
        var dd = d1 - d0
        dst[unsafe_offset=base + i] = (
            -d1 / (cd * d0) * f[unsafe_offset=base + i - 1]
            + dd / (d0 * d1) * f[unsafe_offset=base + i]
            + d0 / (cd * d1) * f[unsafe_offset=base + i + 1]
        )
        i += 1
    d0 = delta[unsafe_offset=(n - 3)]
    d1 = delta[unsafe_offset=(n - 2)]
    cd = d0 + d1
    bd = cd + d1
    dst[unsafe_offset=base + n - 1] = (
        d1 / (cd * d0) * f[unsafe_offset=base + n - 3]
        - cd / (d0 * d1) * f[unsafe_offset=base + n - 2]
        + bd / (cd * d1) * f[unsafe_offset=base + n - 1]
    )



def second_derivative(f: Ptr, delta: Ptr, n: Int, stride: Int, outer: Int, dst: Ptr):
    # see `first_derivative`: only `stride == 1` has a contiguous `W`-lane pass
    var o = 0
    if stride == 1:
        while o < outer:
            _sd_line_v(f, delta, n, o, dst)
            o += 1
    else:
        while o < outer:
            _sd_line_s(f, delta, n, stride, o, dst)
            o += 1


def _sd_line_s(
    f: Ptr, delta: Ptr, n: Int, stride: Int, o: Int, dst: Ptr
):
    var base = (o // stride) * n * stride + (o % stride)
    var d0 = delta[unsafe_offset=(0)]
    var d1 = delta[unsafe_offset=(1)]
    var cd = d0 + d1
    dst[unsafe_offset=base] = 2.0 * (
        f[unsafe_offset=base] / (cd * d0)
        - f[unsafe_offset=base + stride] / (d0 * d1)
        + f[unsafe_offset=base + 2 * stride] / (cd * d1)
    )
    for i in range(1, n - 1):
        d0 = delta[unsafe_offset=(i - 1)]
        d1 = delta[unsafe_offset=(i)]
        cd = d0 + d1
        dst[unsafe_offset=base + i * stride] = 2.0 * (
            f[unsafe_offset=base + (i - 1) * stride] / (cd * d0)
            - f[unsafe_offset=base + i * stride] / (d0 * d1)
            + f[unsafe_offset=base + (i + 1) * stride] / (cd * d1)
        )
    d0 = delta[unsafe_offset=(n - 3)]
    d1 = delta[unsafe_offset=(n - 2)]
    cd = d0 + d1
    dst[unsafe_offset=base + (n - 1) * stride] = 2.0 * (
        f[unsafe_offset=base + (n - 3) * stride] / (cd * d0)
        - f[unsafe_offset=base + (n - 2) * stride] / (d0 * d1)
        + f[unsafe_offset=base + (n - 1) * stride] / (cd * d1)
    )


def _sd_line_v(f: Ptr, delta: Ptr, n: Int, o: Int, dst: Ptr):
    var base = o * n
    var d0 = delta[unsafe_offset=(0)]
    var d1 = delta[unsafe_offset=(1)]
    var cd = d0 + d1
    dst[unsafe_offset=base] = 2.0 * (
        f[unsafe_offset=base] / (cd * d0)
        - f[unsafe_offset=base + 1] / (d0 * d1)
        + f[unsafe_offset=base + 2] / (cd * d1)
    )
    var i = 1
    while i + W <= n - 1:
        var d0v = delta.unsafe_load[width=W]((i - 1))
        var d1v = delta.unsafe_load[width=W]((i))
        var cv = d0v + d1v
        dst.unsafe_store(
            base + i,
            F(2.0)
            * (
                f.unsafe_load[width=W](base + i - 1) / (cv * d0v)
                - f.unsafe_load[width=W](base + i) / (d0v * d1v)
                + f.unsafe_load[width=W](base + i + 1) / (cv * d1v)
            ),
        )
        i += W
    while i < n - 1:
        d0 = delta[unsafe_offset=(i - 1)]
        d1 = delta[unsafe_offset=(i)]
        cd = d0 + d1
        dst[unsafe_offset=base + i] = 2.0 * (
            f[unsafe_offset=base + i - 1] / (cd * d0)
            - f[unsafe_offset=base + i] / (d0 * d1)
            + f[unsafe_offset=base + i + 1] / (cd * d1)
        )
        i += 1
    d0 = delta[unsafe_offset=(n - 3)]
    d1 = delta[unsafe_offset=(n - 2)]
    cd = d0 + d1
    dst[unsafe_offset=base + n - 1] = 2.0 * (
        f[unsafe_offset=base + n - 3] / (cd * d0)
        - f[unsafe_offset=base + n - 2] / (d0 * d1)
        + f[unsafe_offset=base + n - 1] / (cd * d1)
    )



# ---- smoothing (basic.py:860) ----

# Upstream writes only the interior `pad = (n - 1) // 2` region and leaves the
# surrounding edge at its original value, and it evaluates the whole weighted
# sum from the un-updated data before writing back.  Both need a separate
# destination buffer seeded from the input.  The weighted sum is a `W`-lane
# pass along the row with the same (ky, kx) accumulation order as upstream, so
# it is bit for bit the scalar sum.
def smooth_window(
    data: Ptr, dst: Ptr, weights: Ptr, wy: Int, wx: Int, nrows: Int, ncols: Int, passes: Int
):
    var pad_y = (wy - 1) // 2
    var pad_x = (wx - 1) // 2
    _copy(data, dst, nrows * ncols)
    for _ in range(passes):
        for r in range(pad_y, nrows - pad_y):
            var c = pad_x
            # the lanes run along the row: a `W`-lane load gathers the `W` taps of
            # one window element, one window row per lane, so the (ky, kx)
            # accumulation order is upstream's and the sum is unchanged.  The
            # weight is broadcast across the lanes, so this is only the same sum
            # when the window height is the lane width; every other shape falls
            # through to the scalar loop.
            if wy == W:
                while c + W <= ncols - pad_x:
                    var total = F(0.0)
                    for ky in range(wy):
                        for kx in range(wx):
                            total += F(weights[unsafe_offset=ky * wx + kx]) * data.unsafe_load[
                                width=W
                            ]((r + ky - pad_y) * ncols + c + kx - pad_x)
                    dst.unsafe_store(r * ncols + c, total)
                    c += W
            while c < ncols - pad_x:
                var total = 0.0
                for ky in range(wy):
                    for kx in range(wx):
                        total += weights[unsafe_offset=ky * wx + kx] * data[
                            unsafe_offset=(r + ky - pad_y) * ncols + (c + kx - pad_x)
                        ]
                dst[unsafe_offset=r * ncols + c] = total
                c += 1
        _copy(dst, data, nrows * ncols)


def _copy(src: Ptr, dst: Ptr, n: Int):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(i, src.unsafe_load[width=W](i))
        i += W
    while i < n:
        dst.unsafe_store(i, src.unsafe_load(i))
        i += 1


def normalize_window(window: Ptr, m: Int):
    # the sum stays sequential: it is over the window, not over the data, and a
    # tree reduction would move the normalised weights in the last bit
    var total = 0.0
    for k in range(m):
        total += window[unsafe_offset=k]
    var i = 0
    while i + W <= m:
        window.unsafe_store(i, window.unsafe_load[width=W](i) / F(total))
        i += W
    while i < m:
        window[unsafe_offset=i] = window[unsafe_offset=i] / total
        i += 1
