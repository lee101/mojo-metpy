"""Port of `metpy/calc/basic.py` (MetPy 1.7.1), in upstream source order.

`windchill`, `heat_index` and `apparent_temperature` reproduce the branch
structure of the upstream piecewise definitions, including the masked regions.
`mask_undefined` arrives as an `Int`; the mask itself is written to a second
buffer so the Python side can rebuild a `numpy.ma.MaskedArray` exactly as
upstream does. Temperature units are the ones upstream converts to internally
(degC for `windchill`, degF for `heat_index`) and speeds are km/hr.
"""

from std.math import abs, atan2, cos, exp, hypot, isnan, nan, sin, sqrt
from mojometpy.constants import Rd, Re, g, gamma, omega, p0, t0
from max.algorithm import parallelize
from mojometpy.simd import F, PAR_MIN, PAR_WORKERS, W, flag, vpowc
from mojometpy.types import Ptr

comptime EXNER: Float64 = Rd * gamma / g
comptime INVERSE_EXNER: Float64 = g / (Rd * gamma)
comptime RAD2DEG: Float64 = 180.0 / 3.141592653589793


# ---- wind (basic.py:36) ----


def _wind_speed_chunk(
    u: Ptr,
    v: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    # upstream's `np.hypot`, as one `vsqrt` of the squares instead of the scalar
    # `hypot`, which is a three-operand `libm` call per element (measured at
    # about seven times the cost of the vector form on 1M elements)
    var i = lo
    while i + W <= hi:
        var a = u.unsafe_load[width=W](i)
        var b = v.unsafe_load[width=W](i)
        dst.unsafe_store(i, sqrt(a * a + b * b))
        i += W
    while i < hi:
        dst[unsafe_offset=i] = hypot(u[unsafe_offset=i], v[unsafe_offset=i])
        i += 1


def wind_speed(
    u: Ptr,
    v: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _wind_speed_chunk(u, v, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _wind_speed_chunk(u, v, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _wind_direction_chunk(
    u: Ptr,
    v: Ptr,
    convention: Int,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    # convention 0 = 'from', 1 = 'to'; validated in the Python wrapper.
    var i = lo
    while i + W <= n:
        # `units.Quantity(90.0, 'deg') - np.arctan2(-v, -u)`: 90 degrees minus
        # the atan2 result, which upstream's pint subtraction converts from
        # radians.
        var a = -v.unsafe_load[width=W](i)
        var b = -u.unsafe_load[width=W](i)
        var wdir = F(90.0) - atan2(a, b) * F(RAD2DEG)
        if convention == 1:
            wdir -= F(180.0)
        var wdir_masked = wdir + F(360.0) * flag(SIMD.le(wdir, F(0.0)))
        # upstream's `np.ma.masked_where` on a calm wind
        var calm_mask = SIMD.eq(u.unsafe_load[width=W](i), F(0.0)) & SIMD.eq(
            v.unsafe_load[width=W](i), F(0.0)
        )
        dst.unsafe_store(i, calm_mask.select(F(0.0), wdir_masked))
        i += W
    while i < n:
        var wdir = 90.0 - atan2(-v[unsafe_offset=i], -u[unsafe_offset=i]) * RAD2DEG
        if convention == 1:
            wdir -= 180.0
        var mask = wdir <= 0
        if mask:
            wdir += 360.0
        var calm_mask = u[unsafe_offset=i] == 0.0 and v[unsafe_offset=i] == 0.0
        if calm_mask:
            wdir = 0.0
        dst[unsafe_offset=i] = wdir
        i += 1


def wind_direction(
    u: Ptr,
    v: Ptr,
    convention: Int,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _wind_direction_chunk(u, v, convention, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _wind_direction_chunk(u, v, convention, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _wind_components_chunk(
    speed: Ptr,
    wind_direction: Ptr,
    lo: Int,
    hi: Int,
    out_u: Ptr,
    out_v: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        out_u.unsafe_store(i, -speed.unsafe_load[width=W](i) * sin(wind_direction.unsafe_load[width=W](i)))
        out_v.unsafe_store(i, -speed.unsafe_load[width=W](i) * cos(wind_direction.unsafe_load[width=W](i)))
        i += W
    while i < n:
        var direction = wind_direction[unsafe_offset=i]
        out_u[unsafe_offset=i] = -speed[unsafe_offset=i] * sin(direction)
        out_v[unsafe_offset=i] = -speed[unsafe_offset=i] * cos(direction)
        i += 1


def wind_components(
    speed: Ptr,
    wind_direction: Ptr,
    n: Int,
    out_u: Ptr,
    out_v: Ptr,
):
    if n < PAR_MIN:
        _wind_components_chunk(speed, wind_direction, 0, n, out_u, out_v)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _wind_components_chunk(speed, wind_direction, k * span, min((k + 1) * span, n), out_u, out_v)
        parallelize(work, nw, nw)


def _windchill_chunk(
    temperature: Ptr,
    speed: Ptr,
    face_level_winds: Int,
    mask_undefined: Int,
    lo: Int,
    hi: Int,
    dst: Ptr,
    out_mask: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        var s = speed.unsafe_load[width=W](i)
        if face_level_winds == 1:
            s = s * F(1.5)
        var speed_factor = vpowc(s, 0.16)
        var t = temperature.unsafe_load[width=W](i)
        dst.unsafe_store(
            i, (F(0.6215) + F(0.3965) * speed_factor) * t - F(11.37) * speed_factor + F(13.12)
        )
        # upstream's `speed <= units.Quantity(3, 'mph')`, with speed in km/hr
        if mask_undefined == 1:
            out_mask.unsafe_store(
                i, flag(SIMD.gt(t, F(10.0)) | SIMD.le(s, F(4.828032)))
            )
        else:
            out_mask.unsafe_store(i, F(0.0))
        i += W
    while i < n:
        var s = speed[unsafe_offset=i]
        if face_level_winds == 1:
            s = s * 1.5
        var speed_factor = s**0.16
        var t = temperature[unsafe_offset=i]
        dst[unsafe_offset=i] = (0.6215 + 0.3965 * speed_factor) * t - 11.37 * speed_factor + 13.12
        if mask_undefined == 1:
            out_mask[unsafe_offset=i] = 1.0 if (t > 10.0 or s <= 4.828032) else 0.0
        else:
            out_mask[unsafe_offset=i] = 0.0
        i += 1


def windchill(
    temperature: Ptr,
    speed: Ptr,
    face_level_winds: Int,
    mask_undefined: Int,
    n: Int,
    dst: Ptr,
    out_mask: Ptr,
):
    if n < PAR_MIN:
        _windchill_chunk(temperature, speed, face_level_winds, mask_undefined, 0, n, dst, out_mask)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _windchill_chunk(temperature, speed, face_level_winds, mask_undefined, k * span, min((k + 1) * span, n), dst, out_mask)
        parallelize(work, nw, nw)


def _heat_index_chunk(
    temperature: Ptr,
    relative_humidity: Ptr,
    mask_undefined: Int,
    lo: Int,
    hi: Int,
    dst: Ptr,
    out_mask: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        var t = temperature.unsafe_load[width=W](i)
        var rh = relative_humidity.unsafe_load[width=W](i)
        var delta = t - F(0.0)
        var rh2 = rh * rh
        var delta2 = delta * delta
        var a = F(-10.3) + F(1.1) * delta + F(4.7) * rh
        var b = (
            F(-42.379)
            + F(2.04901523) * delta
            + F(1014.333127) * rh
            - F(22.475541) * delta * rh
            - F(0.00683783) * delta2
            - F(548.1717) * rh2
            + F(0.122874) * delta2 * rh
            + F(8.5282) * delta * rh2
            - F(0.0199) * delta2 * rh2
        )
        # upstream's `np.where(t <= 40, t, np.where(a < 79, a, b))`, the inner
        # branch only reached where the outer one is false
        var hi = SIMD.le(t, F(40.0)).select(t, SIMD.lt(a, F(79.0)).select(a, b))
        var rh15 = SIMD.le(rh, F(0.13)) & SIMD.ge(t, F(80.0)) & SIMD.le(t, F(112.0))
        var rh15adj = (F(13.0) - rh * F(100.0)) / F(4.0) * sqrt(
            (F(17.0) - abs(delta - F(95.0))) / F(17.0)
        )
        hi = rh15.select(hi - rh15adj, hi)
        var rh85 = SIMD.gt(rh, F(0.85)) & SIMD.ge(t, F(80.0)) & SIMD.le(t, F(87.0))
        var rh85adj = F(0.02) * (rh * F(100.0) - F(85.0)) * (F(87.0) - delta)
        hi = rh85.select(hi + rh85adj, hi)
        dst.unsafe_store(i, hi)
        if mask_undefined == 1:
            out_mask.unsafe_store(i, flag(SIMD.lt(t, F(80.0))))
        else:
            out_mask.unsafe_store(i, F(0.0))
        i += W
    while i < n:
        var t = temperature[unsafe_offset=i]
        var rh = relative_humidity[unsafe_offset=i]
        var delta = t - 0.0
        var rh2 = rh**2
        var delta2 = delta**2
        var a = -10.3 + 1.1 * delta + 4.7 * rh
        var b = (
            -42.379
            + 2.04901523 * delta
            + 1014.333127 * rh
            - 22.475541 * delta * rh
            - 0.00683783 * delta2
            - 548.1717 * rh2
            + 0.122874 * delta2 * rh
            + 8.5282 * delta * rh2
            - 0.0199 * delta2 * rh2
        )
        var hi = nan[DType.float64]()
        if t <= 40.0:
            hi = t
        if (a < 79.0) and isnan(hi):
            hi = a
        if isnan(hi):
            hi = b
        if rh <= 0.13 and t >= 80.0 and t <= 112.0:
            var rh15adj = (13.0 - rh * 100.0) / 4.0 * sqrt(
                (17.0 - abs(delta - 95.0)) / 17.0
            )
            hi = hi - rh15adj
        if rh > 0.85 and t >= 80.0 and t <= 87.0:
            var rh85adj = 0.02 * (rh * 100.0 - 85.0) * (87.0 - delta)
            hi = hi + rh85adj
        dst[unsafe_offset=i] = hi
        if mask_undefined == 1:
            out_mask[unsafe_offset=i] = 1.0 if t < 80.0 else 0.0
        else:
            out_mask[unsafe_offset=i] = 0.0
        i += 1


def heat_index(
    temperature: Ptr,
    relative_humidity: Ptr,
    mask_undefined: Int,
    n: Int,
    dst: Ptr,
    out_mask: Ptr,
):
    if n < PAR_MIN:
        _heat_index_chunk(temperature, relative_humidity, mask_undefined, 0, n, dst, out_mask)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _heat_index_chunk(temperature, relative_humidity, mask_undefined, k * span, min((k + 1) * span, n), dst, out_mask)
        parallelize(work, nw, nw)


def _apparent_temperature_chunk(
    temperature: Ptr,
    relative_humidity: Ptr,
    speed: Ptr,
    face_level_winds: Int,
    mask_undefined: Int,
    lo: Int,
    hi: Int,
    dst: Ptr,
    out_mask: Ptr,
):
    var n = hi
    # windchill_temperature = windchill(..., mask_undefined=True)
    # heat_index_temperature = heat_index(..., mask_undefined=True)
    # app = where(windchill.mask, heat_index, windchill)
    # The argument is degrees Fahrenheit, the unit `heat_index` works in; the
    # wind chill branch needs `temperature.to(degC)` upstream and is converted
    # back on the way out, so both branches end in the same unit.
    var i = lo
    while i + W <= n:
        var t = temperature.unsafe_load[width=W](i)
        var t_c = (t - F(32.0)) / F(1.8)
        var s = speed.unsafe_load[width=W](i)
        if face_level_winds == 1:
            s = s * F(1.5)
        var speed_factor = vpowc(s, 0.16)
        var wcti_c = (F(0.6215) + F(0.3965) * speed_factor) * t_c - F(11.37) * speed_factor + F(13.12)
        var wcti = wcti_c * F(1.8) + F(32.0)
        var wcti_mask = SIMD.gt(t_c, F(10.0)) | SIMD.le(s, F(4.828032))

        var rh = relative_humidity.unsafe_load[width=W](i)
        var delta = t - F(0.0)
        var rh2 = rh * rh
        var delta2 = delta * delta
        var a = F(-10.3) + F(1.1) * delta + F(4.7) * rh
        var b = (
            F(-42.379)
            + F(2.04901523) * delta
            + F(1014.333127) * rh
            - F(22.475541) * delta * rh
            - F(0.00683783) * delta2
            - F(548.1717) * rh2
            + F(0.122874) * delta2 * rh
            + F(8.5282) * delta * rh2
            - F(0.0199) * delta2 * rh2
        )
        var hi = SIMD.le(t, F(40.0)).select(t, SIMD.lt(a, F(79.0)).select(a, b))
        var rh15 = SIMD.le(rh, F(0.13)) & SIMD.ge(t, F(80.0)) & SIMD.le(t, F(112.0))
        var rh15adj = (F(13.0) - rh * F(100.0)) / F(4.0) * sqrt(
            (F(17.0) - abs(delta - F(95.0))) / F(17.0)
        )
        hi = rh15.select(hi - rh15adj, hi)
        var rh85 = SIMD.gt(rh, F(0.85)) & SIMD.ge(t, F(80.0)) & SIMD.le(t, F(87.0))
        var rh85adj = F(0.02) * (rh * F(100.0) - F(85.0)) * (F(87.0) - delta)
        hi = rh85.select(hi + rh85adj, hi)
        # np.ma.where(wcti.mask, hi, wcti) masks where the selected element is
        # masked, so the composed mask is the intersection of the two.
        var hi_mask = SIMD.lt(t, F(80.0))
        var both = wcti_mask & hi_mask
        var value = wcti_mask.select(hi, wcti)
        if mask_undefined == 0:
            value = both.select(t, value)
            out_mask.unsafe_store(i, F(0.0))
        else:
            out_mask.unsafe_store(i, flag(both))
        dst.unsafe_store(i, value)
        i += W
    while i < n:
        var t = temperature[unsafe_offset=i]
        var t_c = (t - 32.0) / 1.8
        var s = speed[unsafe_offset=i]
        if face_level_winds == 1:
            s = s * 1.5
        var speed_factor = s**0.16
        var wcti_c = (0.6215 + 0.3965 * speed_factor) * t_c - 11.37 * speed_factor + 13.12
        var wcti = wcti_c * 1.8 + 32.0
        var wcti_mask = t_c > 10.0 or s <= 4.828032

        var rh = relative_humidity[unsafe_offset=i]
        var delta = t - 0.0
        var rh2 = rh**2
        var delta2 = delta**2
        var a = -10.3 + 1.1 * delta + 4.7 * rh
        var b = (
            -42.379
            + 2.04901523 * delta
            + 1014.333127 * rh
            - 22.475541 * delta * rh
            - 0.00683783 * delta2
            - 548.1717 * rh2
            + 0.122874 * delta2 * rh
            + 8.5282 * delta * rh2
            - 0.0199 * delta2 * rh2
        )
        var hi = nan[DType.float64]()
        if t <= 40.0:
            hi = t
        if (a < 79.0) and isnan(hi):
            hi = a
        if isnan(hi):
            hi = b
        if rh <= 0.13 and t >= 80.0 and t <= 112.0:
            var rh15adj = (13.0 - rh * 100.0) / 4.0 * sqrt(
                (17.0 - abs(delta - 95.0)) / 17.0
            )
            hi = hi - rh15adj
        if rh > 0.85 and t >= 80.0 and t <= 87.0:
            var rh85adj = 0.02 * (rh * 100.0 - 85.0) * (87.0 - delta)
            hi = hi + rh85adj
        # np.ma.where(wcti.mask, hi, wcti) masks where the selected element is
        # masked, so the composed mask is the intersection of the two.
        var hi_mask = t < 80.0

        var value = wcti
        if wcti_mask:
            value = hi
        if mask_undefined == 0 and wcti_mask and hi_mask:
            value = t
        dst[unsafe_offset=i] = value
        if mask_undefined == 1 and wcti_mask and hi_mask:
            out_mask[unsafe_offset=i] = 1.0
        else:
            out_mask[unsafe_offset=i] = 0.0
        i += 1


def apparent_temperature(
    temperature: Ptr,
    relative_humidity: Ptr,
    speed: Ptr,
    face_level_winds: Int,
    mask_undefined: Int,
    n: Int,
    dst: Ptr,
    out_mask: Ptr,
):
    if n < PAR_MIN:
        _apparent_temperature_chunk(temperature, relative_humidity, speed, face_level_winds, mask_undefined, 0, n, dst, out_mask)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _apparent_temperature_chunk(temperature, relative_humidity, speed, face_level_winds, mask_undefined, k * span, min((k + 1) * span, n), dst, out_mask)
        parallelize(work, nw, nw)


# ---- height and pressure (basic.py:444) ----


def _pressure_to_height_std_chunk(
    pressure: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        dst.unsafe_store(
            i,
            F(t0 / gamma)
            * (F(1.0) - vpowc(pressure.unsafe_load[width=W](i) / F(p0), EXNER)),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = t0 / gamma * (
            1 - (pressure[unsafe_offset=i] / p0) ** EXNER
        )
        i += 1


def pressure_to_height_std(
    pressure: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _pressure_to_height_std_chunk(pressure, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _pressure_to_height_std_chunk(pressure, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _height_to_geopotential_chunk(
    height: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        var h = height.unsafe_load[width=W](i)
        dst.unsafe_store(i, F(g * Re) * h / (F(Re) + h))
        i += W
    while i < n:
        var h = height[unsafe_offset=i]
        dst[unsafe_offset=i] = g * Re * h / (Re + h)
        i += 1


def height_to_geopotential(
    height: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _height_to_geopotential_chunk(height, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _height_to_geopotential_chunk(height, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _geopotential_to_height_chunk(
    geopotential: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        var z = geopotential.unsafe_load[width=W](i)
        dst.unsafe_store(i, z * F(Re) / (F(g * Re) - z))
        i += W
    while i < n:
        var z = geopotential[unsafe_offset=i]
        dst[unsafe_offset=i] = z * Re / (g * Re - z)
        i += 1


def geopotential_to_height(
    geopotential: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _geopotential_to_height_chunk(geopotential, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _geopotential_to_height_chunk(geopotential, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _height_to_pressure_std_chunk(
    height: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        dst.unsafe_store(
            i,
            F(p0)
            * vpowc(
                F(1.0) - F(gamma / t0) * height.unsafe_load[width=W](i),
                INVERSE_EXNER,
            ),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = p0 * (
            1 - gamma / t0 * height[unsafe_offset=i]
        ) ** (
            g / (Rd * gamma)
        )
        i += 1


def height_to_pressure_std(
    height: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _height_to_pressure_std_chunk(height, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _height_to_pressure_std_chunk(height, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _coriolis_parameter_chunk(
    latitude: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    for i in range(lo, hi):
        dst[unsafe_offset=i] = 2.0 * omega * sin(latitude[unsafe_offset=i])


def coriolis_parameter(
    latitude: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _coriolis_parameter_chunk(latitude, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _coriolis_parameter_chunk(latitude, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _add_height_to_pressure_chunk(
    pressure: Ptr,
    height: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        var pressure_level_height = F(t0 / gamma) * (
            F(1.0) - vpowc(pressure.unsafe_load[width=W](i) / F(p0), EXNER)
        )
        dst.unsafe_store(
            i,
            F(p0)
            * vpowc(
                F(1.0)
                - F(gamma / t0)
                * (pressure_level_height + height.unsafe_load[width=W](i)),
                INVERSE_EXNER,
            ),
        )
        i += W
    while i < n:
        var pressure_level_height = t0 / gamma * (
            1 - (pressure[unsafe_offset=i] / p0) ** EXNER
        )
        dst[unsafe_offset=i] = p0 * (
            1 - gamma / t0 * (pressure_level_height + height[unsafe_offset=i])
        ) ** INVERSE_EXNER
        i += 1


def add_height_to_pressure(
    pressure: Ptr,
    height: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _add_height_to_pressure_chunk(pressure, height, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _add_height_to_pressure_chunk(pressure, height, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _add_pressure_to_height_chunk(
    height: Ptr,
    pressure: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        var pressure_at_height = F(p0) * vpowc(
            F(1.0) - F(gamma / t0) * height.unsafe_load[width=W](i), INVERSE_EXNER
        )
        dst.unsafe_store(
            i,
            F(t0 / gamma)
            * (
                F(1.0)
                - vpowc(
                    (pressure_at_height - pressure.unsafe_load[width=W](i)) / F(p0), EXNER
                )
            ),
        )
        i += W
    while i < n:
        var pressure_at_height = p0 * (
            1 - gamma / t0 * height[unsafe_offset=i]
        ) ** INVERSE_EXNER
        dst[unsafe_offset=i] = t0 / gamma * (
            1 - ((pressure_at_height - pressure[unsafe_offset=i]) / p0) ** EXNER
        )
        i += 1


def add_pressure_to_height(
    height: Ptr,
    pressure: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _add_pressure_to_height_chunk(height, pressure, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _add_pressure_to_height_chunk(height, pressure, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _sigma_to_pressure_chunk(
    sigma: Ptr,
    pressure_sfc: Ptr,
    pressure_top: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        var top = pressure_top.unsafe_load[width=W](i)
        dst.unsafe_store(
            i,
            sigma.unsafe_load[width=W](i)
            * (pressure_sfc.unsafe_load[width=W](i) - top)
            + top,
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = sigma[unsafe_offset=i] * (
            pressure_sfc[unsafe_offset=i] - pressure_top[unsafe_offset=i]
        ) + pressure_top[unsafe_offset=i]
        i += 1


def sigma_to_pressure(
    sigma: Ptr,
    pressure_sfc: Ptr,
    pressure_top: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _sigma_to_pressure_chunk(sigma, pressure_sfc, pressure_top, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _sigma_to_pressure_chunk(sigma, pressure_sfc, pressure_top, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)
# ---- altimeter settings (basic.py:1150) ----


def _altimeter_to_station_pressure(altimeter_value: Float64, height: Float64) -> Float64:
    var exponent = Rd * gamma / g
    return (
        altimeter_value**exponent
        - p0**exponent * gamma * height / t0
    ) ** (1 / exponent) + 30.0


def _altimeter_to_station_pressure_v(altimeter_value: F, height: F) -> F:
    return vpowc(
        vpowc(altimeter_value, EXNER)
        - F(p0**EXNER * gamma / t0) * height,
        1 / EXNER,
    ) + F(30.0)


def _altimeter_to_station_pressure_chunk(
    altimeter_value: Ptr,
    height: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        dst.unsafe_store(
            i,
            _altimeter_to_station_pressure_v(
                altimeter_value.unsafe_load[width=W](i), height.unsafe_load[width=W](i)
            ),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = _altimeter_to_station_pressure(
            altimeter_value[unsafe_offset=i], height[unsafe_offset=i]
        )
        i += 1


def altimeter_to_station_pressure(
    altimeter_value: Ptr,
    height: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _altimeter_to_station_pressure_chunk(altimeter_value, height, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _altimeter_to_station_pressure_chunk(altimeter_value, height, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _altimeter_to_sea_level_pressure_chunk(
    altimeter_value: Ptr,
    height: Ptr,
    temperature: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    var i = lo
    while i + W <= n:
        var h = F(Rd) * temperature.unsafe_load[width=W](i) / F(g)
        dst.unsafe_store(
            i,
            _altimeter_to_station_pressure_v(
                altimeter_value.unsafe_load[width=W](i), height.unsafe_load[width=W](i)
            )
            * exp(height.unsafe_load[width=W](i) / h),
        )
        i += W
    while i < n:
        var psfc = _altimeter_to_station_pressure(
            altimeter_value[unsafe_offset=i], height[unsafe_offset=i]
        )
        var h = Rd * temperature[unsafe_offset=i] / g
        dst[unsafe_offset=i] = psfc * exp(height[unsafe_offset=i] / h)
        i += 1


def altimeter_to_sea_level_pressure(
    altimeter_value: Ptr,
    height: Ptr,
    temperature: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _altimeter_to_sea_level_pressure_chunk(altimeter_value, height, temperature, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _altimeter_to_sea_level_pressure_chunk(altimeter_value, height, temperature, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)