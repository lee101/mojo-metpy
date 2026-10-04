"""Port of `metpy/calc/thermo.py` (MetPy 1.7.1), in upstream source order.

Every function keeps upstream's name, argument order and arithmetic; the unit
machinery (pint) is not ported, so all arrays are plain float64 in SI base
units and each `preprocess_and_wrap`/`_nounit` layer collapses to a direct call.
`phase` is the only structural divergence: upstream's `match phase` becomes an
`Int` selector (0 = liquid, 1 = solid, 2 = auto) chosen by the ctypes wrapper.

Each kernel is a `W`-lane vector loop followed by a scalar tail, and each
formula that has a branch is written twice: once for a lane, once for a
scalar.  The kernels whose formula spends a `log` or an `exp` per element also
carry a `_chunk` twin and a `parallelize` split, because they are the only
ones compute bound enough to pay for the launch; the pure arithmetic kernels
are already at the memory bandwidth limit and stay serial.  `lcl` and the
`auto` phase stay scalar because they iterate per element.
"""

from max.algorithm import parallelize
from std.math import abs, exp, log, nan, sqrt
from mojometpy.constants import (
    Cp_d,
    Cp_i,
    Cp_l,
    Cp_v,
    Lf,
    Ls,
    Lv,
    P0,
    Rd,
    Rv,
    T0,
    epsilon,
    g,
    kappa,
    sat_pressure_0c,
    zero_degc,
)
from mojometpy.simd import F, PAR_MIN, PAR_WORKERS, W, div_pos, vpow, vpowc
from mojometpy.types import Ptr

comptime PHASE_LIQUID: Int = 0
comptime PHASE_SOLID: Int = 1
comptime PHASE_AUTO: Int = 2

comptime HEAT_POWER_L: Float64 = (Cp_l - Cp_v) / Rv
comptime HEAT_POWER_S: Float64 = (Cp_i - Cp_v) / Rv


# ---- moist air properties (thermo.py:38) ----

def moist_air_gas_constant(specific_humidity: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(i, Rd + specific_humidity.unsafe_load[width=W](i) * (Rv - Rd))
        i += W
    while i < n:
        dst[unsafe_offset=i] = Rd + specific_humidity[unsafe_offset=i] * (Rv - Rd)
        i += 1


def moist_air_specific_heat_pressure(specific_humidity: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i, Cp_d + specific_humidity.unsafe_load[width=W](i) * (Cp_v - Cp_d)
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = Cp_d + specific_humidity[unsafe_offset=i] * (Cp_v - Cp_d)
        i += 1


def moist_air_poisson_exponent(specific_humidity: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        var q = specific_humidity.unsafe_load[width=W](i)
        dst.unsafe_store(i, (Rd + q * (Rv - Rd)) / (Cp_d + q * (Cp_v - Cp_d)))
        i += W
    while i < n:
        var q = specific_humidity[unsafe_offset=i]
        dst[unsafe_offset=i] = (Rd + q * (Rv - Rd)) / (Cp_d + q * (Cp_v - Cp_d))
        i += 1


# ---- latent heats (thermo.py:153) ----

def water_latent_heat_vaporization(temperature: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            Lv - (Cp_l - Cp_v) * (temperature.unsafe_load[width=W](i) - T0),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = Lv - (Cp_l - Cp_v) * (
            temperature[unsafe_offset=i] - T0
        )
        i += 1


def water_latent_heat_sublimation(temperature: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            Ls - (Cp_i - Cp_v) * (temperature.unsafe_load[width=W](i) - T0),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = Ls - (Cp_i - Cp_v) * (
            temperature[unsafe_offset=i] - T0
        )
        i += 1


def water_latent_heat_melting(temperature: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            Lf - (Cp_l - Cp_i) * (temperature.unsafe_load[width=W](i) - T0),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = Lf - (Cp_l - Cp_i) * (
            temperature[unsafe_offset=i] - T0
        )
        i += 1


# ---- relative humidity and potential temperature (thermo.py:283) ----

def _saturation_vapor_pressure_liquid_v(temperature: F) -> F:
    # the vector twin of `_saturation_vapor_pressure_liquid_1`; the scalar
    # form's two `exp` calls are one, `(T0/t) ** h * exp(e)` being
    # `exp(h*log(T0/t) + e)`
    var latent_heat = Lv - (Cp_l - Cp_v) * (temperature - T0)
    var exp_term = (Lv / T0 - latent_heat / temperature) / Rv
    return F(sat_pressure_0c) * exp(
        log(T0 / temperature) * F(HEAT_POWER_L) + exp_term
    )


def _saturation_vapor_pressure_solid_v(temperature: F) -> F:
    var latent_heat = Ls - (Cp_i - Cp_v) * (temperature - T0)
    var exp_term = (Ls / T0 - latent_heat / temperature) / Rv
    return F(sat_pressure_0c) * exp(
        log(T0 / temperature) * F(HEAT_POWER_S) + exp_term
    )


def _saturation_vapor_pressure_liquid_1(temperature: Float64) -> Float64:
    var latent_heat = Lv - (Cp_l - Cp_v) * (temperature - T0)
    var exp_term = (Lv / T0 - latent_heat / temperature) / Rv
    return sat_pressure_0c * (T0 / temperature) ** HEAT_POWER_L * exp(exp_term)


def _saturation_vapor_pressure_solid_1(temperature: Float64) -> Float64:
    var latent_heat = Ls - (Cp_i - Cp_v) * (temperature - T0)
    var exp_term = (Ls / T0 - latent_heat / temperature) / Rv
    return sat_pressure_0c * (T0 / temperature) ** HEAT_POWER_S * exp(exp_term)


def _saturation_vapor_pressure(temperature: Float64, phase: Int) -> Float64:
    if phase == PHASE_SOLID:
        return _saturation_vapor_pressure_solid_1(temperature)
    elif phase == PHASE_AUTO:
        if temperature > T0:
            return _saturation_vapor_pressure_liquid_1(temperature)
        else:
            return _saturation_vapor_pressure_solid_1(temperature)
    else:
        return _saturation_vapor_pressure_liquid_1(temperature)


def _saturation_mixing_ratio(total_press: Float64, temperature: Float64, phase: Int) -> Float64:
    var e_s = _saturation_vapor_pressure(temperature, phase)
    if e_s >= total_press:
        return nan[DType.float64]()
    return _mixing_ratio(e_s, total_press, epsilon)


def _saturation_mixing_ratio_v(pressure: F, temperature: F) -> F:
    return div_pos(
        F(epsilon) * _saturation_vapor_pressure_liquid_v(temperature),
        pressure - _saturation_vapor_pressure_liquid_v(temperature),
    )


def _virtual_temperature(temperature: Float64, mixing_ratio: Float64, molecular_weight_ratio: Float64) -> Float64:
    return temperature * (
        (mixing_ratio + molecular_weight_ratio)
        / (molecular_weight_ratio * (1 + mixing_ratio))
    )


def _virtual_temperature_v(
    temperature: F, mixing_ratio: F, molecular_weight_ratio: Float64
) -> F:
    return temperature * (
        (mixing_ratio + F(molecular_weight_ratio))
        / (F(molecular_weight_ratio) * (F(1.0) + mixing_ratio))
    )


def _relative_humidity_from_w(w: Float64, w_s: Float64) -> Float64:
    return w / (epsilon + w) * (epsilon + w_s) / w_s


def _relative_humidity_from_dewpoint_chunk(
    temperature: Ptr, dewpoint: Ptr, phase: Int, lo: Int, hi: Int, dst: Ptr
):
    if phase == PHASE_LIQUID:
        var i = lo
        while i + W <= hi:
            var e = _saturation_vapor_pressure_liquid_v(dewpoint.unsafe_load[width=W](i))
            var e_s = _saturation_vapor_pressure_liquid_v(
                temperature.unsafe_load[width=W](i)
            )
            dst.unsafe_store(i, e / e_s)
            i += W
        while i < hi:
            var e = _saturation_vapor_pressure_liquid_1(dewpoint[unsafe_offset=i])
            var e_s = _saturation_vapor_pressure_liquid_1(temperature[unsafe_offset=i])
            dst[unsafe_offset=i] = e / e_s
            i += 1
    else:
        for i in range(lo, hi):
            var e = _saturation_vapor_pressure(dewpoint[unsafe_offset=i], phase)
            var e_s = _saturation_vapor_pressure(temperature[unsafe_offset=i], phase)
            dst[unsafe_offset=i] = e / e_s


def relative_humidity_from_dewpoint(
    temperature: Ptr, dewpoint: Ptr, phase: Int, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _relative_humidity_from_dewpoint_chunk(temperature, dewpoint, phase, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _relative_humidity_from_dewpoint_chunk(
                temperature, dewpoint, phase, k * span, min((k + 1) * span, n), dst
            )
        parallelize(work, nw, nw)


def _exner_function_chunk(
    pressure: Ptr, reference_pressure: Float64, lo: Int, hi: Int, dst: Ptr
):
    var i = lo
    while i + W <= hi:
        dst.unsafe_store(
            i, vpowc(pressure.unsafe_load[width=W](i) / reference_pressure, kappa)
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = (
            pressure[unsafe_offset=i] / reference_pressure
        ) ** kappa
        i += 1


def exner_function(pressure: Ptr, reference_pressure: Float64, n: Int, dst: Ptr):
    if n < PAR_MIN:
        _exner_function_chunk(pressure, reference_pressure, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _exner_function_chunk(
                pressure, reference_pressure, k * span, min((k + 1) * span, n), dst
            )
        parallelize(work, nw, nw)


def _potential_temperature_chunk(
    pressure: Ptr, temperature: Ptr, lo: Int, hi: Int, dst: Ptr
):
    var i = lo
    while i + W <= hi:
        dst.unsafe_store(
            i,
            temperature.unsafe_load[width=W](i)
            / vpowc(pressure.unsafe_load[width=W](i) / F(P0), kappa),
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = temperature[unsafe_offset=i] / (
            pressure[unsafe_offset=i] / P0
        ) ** kappa
        i += 1


def potential_temperature(pressure: Ptr, temperature: Ptr, n: Int, dst: Ptr):
    if n < PAR_MIN:
        _potential_temperature_chunk(pressure, temperature, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _potential_temperature_chunk(
                pressure, temperature, k * span, min((k + 1) * span, n), dst
            )
        parallelize(work, nw, nw)


def _temperature_from_potential_temperature_chunk(
    pressure: Ptr, potential_temperature: Ptr, lo: Int, hi: Int, dst: Ptr
):
    var i = lo
    while i + W <= hi:
        dst.unsafe_store(
            i,
            potential_temperature.unsafe_load[width=W](i)
            * vpowc(pressure.unsafe_load[width=W](i) / F(P0), kappa),
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = potential_temperature[unsafe_offset=i] * (
            pressure[unsafe_offset=i] / P0
        ) ** kappa
        i += 1


def temperature_from_potential_temperature(
    pressure: Ptr, potential_temperature: Ptr, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _temperature_from_potential_temperature_chunk(
            pressure, potential_temperature, 0, n, dst
        )
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _temperature_from_potential_temperature_chunk(
                pressure, potential_temperature, k * span, min((k + 1) * span, n), dst
            )
        parallelize(work, nw, nw)


def _dry_lapse_chunk(
    pressure: Ptr, temperature: Ptr, reference_pressure: Ptr, lo: Int, hi: Int, dst: Ptr
):
    var i = lo
    while i + W <= hi:
        dst.unsafe_store(
            i,
            temperature.unsafe_load[width=W](i)
            * vpow(
                pressure.unsafe_load[width=W](i)
                / reference_pressure.unsafe_load[width=W](i),
                F(kappa),
            ),
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = temperature[unsafe_offset=i] * (
            pressure[unsafe_offset=i] / reference_pressure[unsafe_offset=i]
        ) ** kappa
        i += 1


def dry_lapse(
    pressure: Ptr, temperature: Ptr, reference_pressure: Ptr, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _dry_lapse_chunk(pressure, temperature, reference_pressure, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _dry_lapse_chunk(
                pressure,
                temperature,
                reference_pressure,
                k * span,
                min((k + 1) * span, n),
                dst,
            )
        parallelize(work, nw, nw)


def _saturation_vapor_pressure_liquid(temperature: Ptr, n: Int, dst: Ptr):
    saturation_vapor_pressure(temperature, PHASE_LIQUID, n, dst)


def _saturation_vapor_pressure_solid(temperature: Ptr, n: Int, dst: Ptr):
    saturation_vapor_pressure(temperature, PHASE_SOLID, n, dst)


# ---- lifting condensation level (thermo.py:663) ----

# The real Lambert W branch k = -1: solve w*exp(w) = x for w <= -1, which is
# real only on [-1/e, 0).  With u = -w >= 1 the equation becomes
# u - log(u) = -log(-x); the left side is strictly increasing for u >= 1, so a
# bracketed Newton solve cannot diverge.
def _lambertw_m1(x: Float64) -> Float64:
    if x < -0.36787944117144233 or x >= 0.0:  # outside [-1/e, 0)
        return nan[DType.float64]()
    var log_t = log(-x)
    # bracket the root u >= 1 of u - log(u) + log(-x) = 0 by doubling
    var u = 1.0
    while (u - log(u) + log_t) < 0.0 and u < 1.0e12:
        u = u * 2.0
    # 1 - 1/u is strictly positive for u >= 1, so Newton started at the upper
    # end of the bracket converges monotonically and cannot diverge
    for _ in range(200):
        var step = (u - log(u) + log_t) / (1.0 - 1.0 / u)
        u = u - step
        if abs(step) < 1.0e-15 * (1.0 + abs(u)):
            break
    return -u


def _lcl_chunk(
    pressure: Ptr,
    temperature: Ptr,
    dewpoint: Ptr,
    lo: Int,
    hi: Int,
    out_pressure: Ptr,
    out_temperature: Ptr,
):
    for i in range(lo, hi):
        var p = pressure[unsafe_offset=i]
        var t = temperature[unsafe_offset=i]
        var td = dewpoint[unsafe_offset=i]
        var q = _specific_humidity_from_dewpoint(p, td, PHASE_LIQUID)
        var moist_heat_ratio = _moist_air_specific_heat_pressure(q) / _moist_air_gas_constant(q)
        var spec_heat_diff = Cp_l - Cp_v
        var a = moist_heat_ratio + spec_heat_diff / Rv
        var b = -(Lv + spec_heat_diff * T0) / (Rv * t)
        var c = b / a
        var rh = _relative_humidity_from_dewpoint(t, td, PHASE_LIQUID)
        var w_minus1 = _lambertw_m1(rh ** (1.0 / a) * c * exp(c))
        var t_lcl = c / w_minus1 * t
        out_temperature[unsafe_offset=i] = t_lcl
        out_pressure[unsafe_offset=i] = p * (t_lcl / t) ** moist_heat_ratio


def lcl(pressure: Ptr, temperature: Ptr, dewpoint: Ptr, n: Int, out_pressure: Ptr, out_temperature: Ptr):
    if n < PAR_MIN:
        _lcl_chunk(
            pressure, temperature, dewpoint, 0, n, out_pressure, out_temperature
        )
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _lcl_chunk(
                pressure,
                temperature,
                dewpoint,
                k * span,
                min((k + 1) * span, n),
                out_pressure,
                out_temperature,
            )
        parallelize(work, nw, nw)


# ---- vapor pressure and mixing ratio (thermo.py:1491) ----

def vapor_pressure(pressure: Ptr, mixing_ratio: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        var w = mixing_ratio.unsafe_load[width=W](i)
        dst.unsafe_store(i, pressure.unsafe_load[width=W](i) * w / (F(epsilon) + w))
        i += W
    while i < n:
        var w = mixing_ratio[unsafe_offset=i]
        dst[unsafe_offset=i] = pressure[unsafe_offset=i] * w / (epsilon + w)
        i += 1


def _saturation_vapor_pressure_liquid_chunk(temperature: Ptr, lo: Int, hi: Int, dst: Ptr):
    var i = lo
    while i + W <= hi:
        dst.unsafe_store(
            i, _saturation_vapor_pressure_liquid_v(temperature.unsafe_load[width=W](i))
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _saturation_vapor_pressure_liquid_1(
            temperature[unsafe_offset=i]
        )
        i += 1


def _saturation_vapor_pressure_solid_chunk(temperature: Ptr, lo: Int, hi: Int, dst: Ptr):
    var i = lo
    while i + W <= hi:
        dst.unsafe_store(
            i, _saturation_vapor_pressure_solid_v(temperature.unsafe_load[width=W](i))
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _saturation_vapor_pressure_solid_1(
            temperature[unsafe_offset=i]
        )
        i += 1


def saturation_vapor_pressure(temperature: Ptr, phase: Int, n: Int, dst: Ptr):
    if phase == PHASE_AUTO:
        for i in range(n):
            dst[unsafe_offset=i] = _saturation_vapor_pressure(
                temperature[unsafe_offset=i], phase
            )
        return
    var nw = PAR_WORKERS
    var chunk = _saturation_vapor_pressure_liquid_chunk
    if phase == PHASE_SOLID:
        chunk = _saturation_vapor_pressure_solid_chunk
    if n < PAR_MIN:
        chunk(temperature, 0, n, dst)
    else:
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            chunk(temperature, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _dewpoint(vapor_pressure: Float64) -> Float64:
    var val = log(vapor_pressure / sat_pressure_0c)
    return zero_degc + 243.5 * val / (17.67 - val)


def _dewpoint_chunk(vapor_pressure: Ptr, lo: Int, hi: Int, dst: Ptr):
    var i = lo
    while i + W <= hi:
        var val = log(vapor_pressure.unsafe_load[width=W](i) / F(sat_pressure_0c))
        dst.unsafe_store(i, F(zero_degc) + F(243.5) * val / (F(17.67) - val))
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _dewpoint(vapor_pressure[unsafe_offset=i])
        i += 1


def dewpoint(vapor_pressure: Ptr, n: Int, dst: Ptr):
    if n < PAR_MIN:
        _dewpoint_chunk(vapor_pressure, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _dewpoint_chunk(vapor_pressure, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _dewpoint_from_relative_humidity_chunk(
    temperature: Ptr, relative_humidity: Ptr, lo: Int, hi: Int, dst: Ptr
):
    var i = lo
    while i + W <= hi:
        var e = relative_humidity.unsafe_load[width=W](i) * _saturation_vapor_pressure_liquid_v(
            temperature.unsafe_load[width=W](i)
        )
        var val = log(e / F(sat_pressure_0c))
        dst.unsafe_store(i, F(zero_degc) + F(243.5) * val / (F(17.67) - val))
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _dewpoint(
            relative_humidity[unsafe_offset=i]
            * _saturation_vapor_pressure_liquid_1(temperature[unsafe_offset=i])
        )
        i += 1


def dewpoint_from_relative_humidity(
    temperature: Ptr, relative_humidity: Ptr, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _dewpoint_from_relative_humidity_chunk(temperature, relative_humidity, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _dewpoint_from_relative_humidity_chunk(
                temperature, relative_humidity, k * span, min((k + 1) * span, n), dst
            )
        parallelize(work, nw, nw)


def _mixing_ratio(partial_press: Float64, total_press: Float64, molecular_weight_ratio: Float64) -> Float64:
    return molecular_weight_ratio * partial_press / (total_press - partial_press)


def mixing_ratio(
    partial_press: Ptr, total_press: Ptr, molecular_weight_ratio: Float64, n: Int, dst: Ptr
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            F(molecular_weight_ratio)
            * partial_press.unsafe_load[width=W](i)
            / (total_press.unsafe_load[width=W](i) - partial_press.unsafe_load[width=W](i)),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = _mixing_ratio(
            partial_press[unsafe_offset=i],
            total_press[unsafe_offset=i],
            molecular_weight_ratio,
        )
        i += 1


def _saturation_mixing_ratio_chunk(
    total_press: Ptr, temperature: Ptr, phase: Int, lo: Int, hi: Int, dst: Ptr
):
    if phase != PHASE_LIQUID:
        for i in range(lo, hi):
            dst[unsafe_offset=i] = _saturation_mixing_ratio(
                total_press[unsafe_offset=i], temperature[unsafe_offset=i], phase
            )
        return
    var i = lo
    while i + W <= hi:
        dst.unsafe_store(
            i,
            _saturation_mixing_ratio_v(
                total_press.unsafe_load[width=W](i),
                temperature.unsafe_load[width=W](i),
            ),
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _saturation_mixing_ratio(
            total_press[unsafe_offset=i], temperature[unsafe_offset=i], phase
        )
        i += 1


def saturation_mixing_ratio(
    total_press: Ptr, temperature: Ptr, phase: Int, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _saturation_mixing_ratio_chunk(total_press, temperature, phase, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _saturation_mixing_ratio_chunk(
                total_press, temperature, phase, k * span, min((k + 1) * span, n), dst
            )
        parallelize(work, nw, nw)


# ---- equivalent potential temperature (thermo.py:1888) ----

def _equivalent_potential_temperature_chunk(
    pressure: Ptr, temperature: Ptr, dewpoint: Ptr, lo: Int, hi: Int, dst: Ptr
):
    var i = lo
    while i + W <= hi:
        var p = pressure.unsafe_load[width=W](i)
        var t = temperature.unsafe_load[width=W](i)
        var td = dewpoint.unsafe_load[width=W](i)
        var e = _saturation_vapor_pressure_liquid_v(td)
        var r = div_pos(F(epsilon) * e, p - e)
        var t_l = F(56.0) + F(1.0) / (F(1.0) / (td - F(56.0)) + log(t / td) / F(800.0))
        var th_l = (t / vpowc((p - e) / F(P0), kappa)) * vpow(t / t_l, F(0.28) * r)
        dst.unsafe_store(
            i,
            th_l
            * exp(r * (F(1.0) + F(0.448) * r) * (F(3036.0) / t_l - F(1.78))),
        )
        i += W
    while i < hi:
        var p = pressure[unsafe_offset=i]
        var t = temperature[unsafe_offset=i]
        var td = dewpoint[unsafe_offset=i]
        var r = _saturation_mixing_ratio(p, td, PHASE_LIQUID)
        var e = _saturation_vapor_pressure_liquid_1(td)
        var t_l = 56 + 1.0 / (1.0 / (td - 56) + log(t / td) / 800.0)
        var th_l = (t / ((p - e) / P0) ** kappa) * (t / t_l) ** (0.28 * r)
        dst[unsafe_offset=i] = th_l * exp(r * (1 + 0.448 * r) * (3036.0 / t_l - 1.78))
        i += 1


def equivalent_potential_temperature(
    pressure: Ptr, temperature: Ptr, dewpoint: Ptr, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _equivalent_potential_temperature_chunk(pressure, temperature, dewpoint, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _equivalent_potential_temperature_chunk(
                pressure, temperature, dewpoint, k * span, min((k + 1) * span, n), dst
            )
        parallelize(work, nw, nw)


# ---- virtual temperature and density (thermo.py:2085) ----

def virtual_temperature(
    temperature: Ptr, mixing_ratio: Ptr, molecular_weight_ratio: Float64, n: Int, dst: Ptr
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            _virtual_temperature_v(
                temperature.unsafe_load[width=W](i),
                mixing_ratio.unsafe_load[width=W](i),
                molecular_weight_ratio,
            ),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = _virtual_temperature(
            temperature[unsafe_offset=i],
            mixing_ratio[unsafe_offset=i],
            molecular_weight_ratio,
        )
        i += 1


def _virtual_temperature_from_dewpoint_chunk(
    pressure: Ptr,
    temperature: Ptr,
    dewpoint: Ptr,
    molecular_weight_ratio: Float64,
    phase: Int,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    if phase != PHASE_LIQUID:
        for i in range(lo, hi):
            dst[unsafe_offset=i] = _virtual_temperature(
                temperature[unsafe_offset=i],
                _saturation_mixing_ratio(
                    pressure[unsafe_offset=i], dewpoint[unsafe_offset=i], phase
                ),
                molecular_weight_ratio,
            )
        return
    var i = lo
    while i + W <= hi:
        dst.unsafe_store(
            i,
            _virtual_temperature_v(
                temperature.unsafe_load[width=W](i),
                _saturation_mixing_ratio_v(
                    pressure.unsafe_load[width=W](i),
                    dewpoint.unsafe_load[width=W](i),
                ),
                molecular_weight_ratio,
            ),
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _virtual_temperature(
            temperature[unsafe_offset=i],
            _saturation_mixing_ratio(
                pressure[unsafe_offset=i], dewpoint[unsafe_offset=i], phase
            ),
            molecular_weight_ratio,
        )
        i += 1


def virtual_temperature_from_dewpoint(
    pressure: Ptr,
    temperature: Ptr,
    dewpoint: Ptr,
    molecular_weight_ratio: Float64,
    phase: Int,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _virtual_temperature_from_dewpoint_chunk(
            pressure, temperature, dewpoint, molecular_weight_ratio, phase, 0, n, dst
        )
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _virtual_temperature_from_dewpoint_chunk(
                pressure,
                temperature,
                dewpoint,
                molecular_weight_ratio,
                phase,
                k * span,
                min((k + 1) * span, n),
                dst,
            )
        parallelize(work, nw, nw)


def _virtual_potential_temperature_chunk(
    pressure: Ptr,
    temperature: Ptr,
    mixing_ratio: Ptr,
    molecular_weight_ratio: Float64,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var i = lo
    while i + W <= hi:
        var pottemp = temperature.unsafe_load[width=W](i) / vpowc(
            pressure.unsafe_load[width=W](i) / F(P0), kappa
        )
        dst.unsafe_store(
            i,
            _virtual_temperature_v(
                pottemp, mixing_ratio.unsafe_load[width=W](i), molecular_weight_ratio
            ),
        )
        i += W
    while i < hi:
        var pottemp = temperature[unsafe_offset=i] / (
            pressure[unsafe_offset=i] / P0
        ) ** kappa
        dst[unsafe_offset=i] = _virtual_temperature(
            pottemp, mixing_ratio[unsafe_offset=i], molecular_weight_ratio
        )
        i += 1


def virtual_potential_temperature(
    pressure: Ptr,
    temperature: Ptr,
    mixing_ratio: Ptr,
    molecular_weight_ratio: Float64,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _virtual_potential_temperature_chunk(
            pressure, temperature, mixing_ratio, molecular_weight_ratio, 0, n, dst
        )
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _virtual_potential_temperature_chunk(
                pressure,
                temperature,
                mixing_ratio,
                molecular_weight_ratio,
                k * span,
                min((k + 1) * span, n),
                dst,
            )
        parallelize(work, nw, nw)


def _density(pressure: Float64, temperature: Float64, mixing_ratio: Float64, molecular_weight_ratio: Float64) -> Float64:
    var virttemp = _virtual_temperature(temperature, mixing_ratio, molecular_weight_ratio)
    return pressure / (Rd * virttemp)


def density(
    pressure: Ptr,
    temperature: Ptr,
    mixing_ratio: Ptr,
    molecular_weight_ratio: Float64,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            pressure.unsafe_load[width=W](i)
            / (
                F(Rd)
                * _virtual_temperature_v(
                    temperature.unsafe_load[width=W](i),
                    mixing_ratio.unsafe_load[width=W](i),
                    molecular_weight_ratio,
                )
            ),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = _density(
            pressure[unsafe_offset=i],
            temperature[unsafe_offset=i],
            mixing_ratio[unsafe_offset=i],
            molecular_weight_ratio,
        )
        i += 1


# ---- psychrometry (thermo.py:2309) ----

def _psychrometric_vapor_pressure_wet(
    pressure: Float64, dry_bulb_temperature: Float64, wet_bulb_temperature: Float64, psychrometer_coefficient: Float64
) -> Float64:
    return _saturation_vapor_pressure_liquid_1(wet_bulb_temperature) - psychrometer_coefficient * pressure * (
        dry_bulb_temperature - wet_bulb_temperature
    )


def _psychrometric_vapor_pressure_wet_chunk(
    pressure: Ptr,
    dry_bulb_temperature: Ptr,
    wet_bulb_temperature: Ptr,
    psychrometer_coefficient: Float64,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var i = lo
    while i + W <= hi:
        var db = dry_bulb_temperature.unsafe_load[width=W](i)
        var wb = wet_bulb_temperature.unsafe_load[width=W](i)
        dst.unsafe_store(
            i,
            _saturation_vapor_pressure_liquid_v(wb)
            - F(psychrometer_coefficient) * pressure.unsafe_load[width=W](i) * (db - wb),
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _psychrometric_vapor_pressure_wet(
            pressure[unsafe_offset=i],
            dry_bulb_temperature[unsafe_offset=i],
            wet_bulb_temperature[unsafe_offset=i],
            psychrometer_coefficient,
        )
        i += 1


def psychrometric_vapor_pressure_wet(
    pressure: Ptr,
    dry_bulb_temperature: Ptr,
    wet_bulb_temperature: Ptr,
    psychrometer_coefficient: Float64,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _psychrometric_vapor_pressure_wet_chunk(
            pressure, dry_bulb_temperature, wet_bulb_temperature, psychrometer_coefficient, 0, n, dst
        )
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _psychrometric_vapor_pressure_wet_chunk(
                pressure,
                dry_bulb_temperature,
                wet_bulb_temperature,
                psychrometer_coefficient,
                k * span,
                min((k + 1) * span, n),
                dst,
            )
        parallelize(work, nw, nw)


def _relative_humidity_wet_psychrometric_chunk(
    pressure: Ptr,
    dry_bulb_temperature: Ptr,
    wet_bulb_temperature: Ptr,
    psychrometer_coefficient: Float64,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var i = lo
    while i + W <= hi:
        var db = dry_bulb_temperature.unsafe_load[width=W](i)
        var wb = wet_bulb_temperature.unsafe_load[width=W](i)
        dst.unsafe_store(
            i,
            (
                _saturation_vapor_pressure_liquid_v(wb)
                - F(psychrometer_coefficient) * pressure.unsafe_load[width=W](i) * (db - wb)
            )
            / _saturation_vapor_pressure_liquid_v(db),
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _psychrometric_vapor_pressure_wet(
            pressure[unsafe_offset=i],
            dry_bulb_temperature[unsafe_offset=i],
            wet_bulb_temperature[unsafe_offset=i],
            psychrometer_coefficient,
        ) / _saturation_vapor_pressure_liquid_1(dry_bulb_temperature[unsafe_offset=i])
        i += 1


def relative_humidity_wet_psychrometric(
    pressure: Ptr,
    dry_bulb_temperature: Ptr,
    wet_bulb_temperature: Ptr,
    psychrometer_coefficient: Float64,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _relative_humidity_wet_psychrometric_chunk(
            pressure, dry_bulb_temperature, wet_bulb_temperature, psychrometer_coefficient, 0, n, dst
        )
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _relative_humidity_wet_psychrometric_chunk(
                pressure,
                dry_bulb_temperature,
                wet_bulb_temperature,
                psychrometer_coefficient,
                k * span,
                min((k + 1) * span, n),
                dst,
            )
        parallelize(work, nw, nw)


# ---- humidity conversions (thermo.py:2442) ----

def _mixing_ratio_from_relative_humidity(
    pressure: Float64, temperature: Float64, relative_humidity: Float64, phase: Int
) -> Float64:
    var w_s = _saturation_mixing_ratio(pressure, temperature, phase)
    return epsilon * w_s * relative_humidity / (epsilon + w_s * (1 - relative_humidity))


def _mixing_ratio_from_relative_humidity_chunk(
    pressure: Ptr,
    temperature: Ptr,
    relative_humidity: Ptr,
    phase: Int,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    if phase != PHASE_LIQUID:
        for i in range(lo, hi):
            dst[unsafe_offset=i] = _mixing_ratio_from_relative_humidity(
                pressure[unsafe_offset=i],
                temperature[unsafe_offset=i],
                relative_humidity[unsafe_offset=i],
                phase,
            )
        return
    var i = lo
    while i + W <= hi:
        var w_s = _saturation_mixing_ratio_v(
            pressure.unsafe_load[width=W](i), temperature.unsafe_load[width=W](i)
        )
        var rh = relative_humidity.unsafe_load[width=W](i)
        dst.unsafe_store(
            i, F(epsilon) * w_s * rh / (F(epsilon) + w_s * (F(1.0) - rh))
        )
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _mixing_ratio_from_relative_humidity(
            pressure[unsafe_offset=i],
            temperature[unsafe_offset=i],
            relative_humidity[unsafe_offset=i],
            phase,
        )
        i += 1


def mixing_ratio_from_relative_humidity(
    pressure: Ptr, temperature: Ptr, relative_humidity: Ptr, phase: Int, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _mixing_ratio_from_relative_humidity_chunk(
            pressure, temperature, relative_humidity, phase, 0, n, dst
        )
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _mixing_ratio_from_relative_humidity_chunk(
                pressure,
                temperature,
                relative_humidity,
                phase,
                k * span,
                min((k + 1) * span, n),
                dst,
            )
        parallelize(work, nw, nw)


def _relative_humidity_from_mixing_ratio_chunk(
    pressure: Ptr, temperature: Ptr, mixing_ratio: Ptr, phase: Int, lo: Int, hi: Int, dst: Ptr
):
    if phase != PHASE_LIQUID:
        for i in range(lo, hi):
            dst[unsafe_offset=i] = _relative_humidity_from_w(
                mixing_ratio[unsafe_offset=i],
                _saturation_mixing_ratio(
                    pressure[unsafe_offset=i], temperature[unsafe_offset=i], phase
                ),
            )
        return
    var i = lo
    while i + W <= hi:
        var w_s = _saturation_mixing_ratio_v(
            pressure.unsafe_load[width=W](i), temperature.unsafe_load[width=W](i)
        )
        var w = mixing_ratio.unsafe_load[width=W](i)
        dst.unsafe_store(i, w / (F(epsilon) + w) * (F(epsilon) + w_s) / w_s)
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _relative_humidity_from_w(
            mixing_ratio[unsafe_offset=i],
            _saturation_mixing_ratio(
                pressure[unsafe_offset=i], temperature[unsafe_offset=i], phase
            ),
        )
        i += 1


def relative_humidity_from_mixing_ratio(
    pressure: Ptr, temperature: Ptr, mixing_ratio: Ptr, phase: Int, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _relative_humidity_from_mixing_ratio_chunk(
            pressure, temperature, mixing_ratio, phase, 0, n, dst
        )
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _relative_humidity_from_mixing_ratio_chunk(
                pressure,
                temperature,
                mixing_ratio,
                phase,
                k * span,
                min((k + 1) * span, n),
                dst,
            )
        parallelize(work, nw, nw)


def mixing_ratio_from_specific_humidity(specific_humidity: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        var q = specific_humidity.unsafe_load[width=W](i)
        dst.unsafe_store(i, q / (F(1.0) - q))
        i += W
    while i < n:
        dst[unsafe_offset=i] = specific_humidity[unsafe_offset=i] / (
            1 - specific_humidity[unsafe_offset=i]
        )
        i += 1


def specific_humidity_from_mixing_ratio(mixing_ratio: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        var w = mixing_ratio.unsafe_load[width=W](i)
        dst.unsafe_store(i, w / (F(1.0) + w))
        i += W
    while i < n:
        dst[unsafe_offset=i] = mixing_ratio[unsafe_offset=i] / (
            1 + mixing_ratio[unsafe_offset=i]
        )
        i += 1


def _relative_humidity_from_specific_humidity_chunk(
    pressure: Ptr, temperature: Ptr, specific_humidity: Ptr, phase: Int, lo: Int, hi: Int, dst: Ptr
):
    if phase != PHASE_LIQUID:
        for i in range(lo, hi):
            var q = specific_humidity[unsafe_offset=i]
            dst[unsafe_offset=i] = _relative_humidity_from_w(
                q / (1 - q),
                _saturation_mixing_ratio(
                    pressure[unsafe_offset=i], temperature[unsafe_offset=i], phase
                ),
            )
        return
    var i = lo
    while i + W <= hi:
        var q = specific_humidity.unsafe_load[width=W](i)
        var w = q / (F(1.0) - q)
        var w_s = _saturation_mixing_ratio_v(
            pressure.unsafe_load[width=W](i), temperature.unsafe_load[width=W](i)
        )
        dst.unsafe_store(i, w / (F(epsilon) + w) * (F(epsilon) + w_s) / w_s)
        i += W
    while i < hi:
        var q = specific_humidity[unsafe_offset=i]
        dst[unsafe_offset=i] = _relative_humidity_from_w(
            q / (1 - q),
            _saturation_mixing_ratio(
                pressure[unsafe_offset=i], temperature[unsafe_offset=i], phase
            ),
        )
        i += 1


def relative_humidity_from_specific_humidity(
    pressure: Ptr, temperature: Ptr, specific_humidity: Ptr, phase: Int, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _relative_humidity_from_specific_humidity_chunk(
            pressure, temperature, specific_humidity, phase, 0, n, dst
        )
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _relative_humidity_from_specific_humidity_chunk(
                pressure,
                temperature,
                specific_humidity,
                phase,
                k * span,
                min((k + 1) * span, n),
                dst,
            )
        parallelize(work, nw, nw)


# ---- static energy and thickness (thermo.py:3864) ----

def dry_static_energy(height: Ptr, temperature: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            F(g) * height.unsafe_load[width=W](i)
            + F(Cp_d) * temperature.unsafe_load[width=W](i),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = g * height[unsafe_offset=i] + Cp_d * temperature[
            unsafe_offset=i
        ]
        i += 1


def moist_static_energy(
    height: Ptr, temperature: Ptr, specific_humidity: Ptr, n: Int, dst: Ptr
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            F(g) * height.unsafe_load[width=W](i)
            + F(Cp_d) * temperature.unsafe_load[width=W](i)
            + F(Lv) * specific_humidity.unsafe_load[width=W](i),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = (
            g * height[unsafe_offset=i]
            + Cp_d * temperature[unsafe_offset=i]
            + Lv * specific_humidity[unsafe_offset=i]
        )
        i += 1


def thickness_hydrostatic(
    pressure: Ptr,
    temperature: Ptr,
    mixing_ratio: Ptr,
    use_mixing: Int,
    molecular_weight_ratio: Float64,
    n: Int,
    dst: Ptr,
):
    # `use_mixing` carries upstream's `mixing_ratio is None` branch.  It is
    # loop invariant, so the pass is one `W`-lane trapezoid reduction with a
    # single running sum and a scalar tail.
    var acc = F(0.0)
    var total = 0.0
    var i = 0
    while i + W <= n - 1:
        var vp0 = temperature.unsafe_load[width=W](i)
        var vp1 = temperature.unsafe_load[width=W](i + 1)
        if use_mixing == 1:
            vp0 = _virtual_temperature_v(
                vp0, mixing_ratio.unsafe_load[width=W](i), molecular_weight_ratio
            )
            vp1 = _virtual_temperature_v(
                vp1, mixing_ratio.unsafe_load[width=W](i + 1), molecular_weight_ratio
            )
        acc += (
            log(pressure.unsafe_load[width=W](i + 1))
            - log(pressure.unsafe_load[width=W](i))
        ) * (vp1 + vp0) / 2.0
        i += W
    while i < n - 1:
        var vp0 = temperature[unsafe_offset=i]
        var vp1 = temperature[unsafe_offset=i + 1]
        if use_mixing == 1:
            vp0 = _virtual_temperature(
                temperature[unsafe_offset=i],
                mixing_ratio[unsafe_offset=i],
                molecular_weight_ratio,
            )
            vp1 = _virtual_temperature(
                temperature[unsafe_offset=i + 1],
                mixing_ratio[unsafe_offset=i + 1],
                molecular_weight_ratio,
            )
        total += (log(pressure[unsafe_offset=i + 1]) - log(pressure[unsafe_offset=i])) * (
            vp1 + vp0
        ) / 2.0
        i += 1
    dst[unsafe_offset=0] = -Rd / g * (total + acc.reduce_add())


def _virtual_temperature_from_rh(
    pressure: Float64, temperature: Float64, relative_humidity: Float64
) -> Float64:
    return _virtual_temperature(
        temperature,
        _mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity, PHASE_LIQUID),
        epsilon,
    )


def _virtual_temperature_from_rh_v(
    pressure: F, temperature: F, relative_humidity: F
) -> F:
    var e_s = _saturation_vapor_pressure_liquid_v(temperature)
    var w_s = div_pos(F(epsilon) * e_s, pressure - e_s)
    var w = F(epsilon) * w_s * relative_humidity / (
        F(epsilon) + w_s * (F(1.0) - relative_humidity)
    )
    return _virtual_temperature_v(temperature, w, epsilon)


def thickness_hydrostatic_from_relative_humidity(
    pressure: Ptr, temperature: Ptr, relative_humidity: Ptr, n: Int, dst: Ptr
):
    var acc = F(0.0)
    var total = 0.0
    var i = 0
    while i + W <= n - 1:
        acc += (
            log(pressure.unsafe_load[width=W](i + 1))
            - log(pressure.unsafe_load[width=W](i))
        ) * (
            _virtual_temperature_from_rh_v(
                pressure.unsafe_load[width=W](i),
                temperature.unsafe_load[width=W](i),
                relative_humidity.unsafe_load[width=W](i),
            )
            + _virtual_temperature_from_rh_v(
                pressure.unsafe_load[width=W](i + 1),
                temperature.unsafe_load[width=W](i + 1),
                relative_humidity.unsafe_load[width=W](i + 1),
            )
        ) / 2.0
        i += W
    while i < n - 1:
        var vp0 = _virtual_temperature_from_rh(
            pressure[unsafe_offset=i], temperature[unsafe_offset=i], relative_humidity[unsafe_offset=i]
        )
        var vp1 = _virtual_temperature_from_rh(
            pressure[unsafe_offset=i + 1], temperature[unsafe_offset=i + 1], relative_humidity[unsafe_offset=i + 1]
        )
        total += (log(pressure[unsafe_offset=i + 1]) - log(pressure[unsafe_offset=i])) * (
            vp1 + vp0
        ) / 2.0
        i += 1
    dst[unsafe_offset=0] = -Rd / g * (total + acc.reduce_add())


# ---- stability (thermo.py:4204) ----

def brunt_vaisala_frequency_squared(
    height: Ptr, dtheta_dz: Ptr, potential_temperature: Ptr, n: Int, dst: Ptr
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            F(g)
            / potential_temperature.unsafe_load[width=W](i)
            * dtheta_dz.unsafe_load[width=W](i),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = g / potential_temperature[unsafe_offset=i] * dtheta_dz[
            unsafe_offset=i
        ]
        i += 1


def brunt_vaisala_frequency(bv_freq_squared: Ptr, n: Int, dst: Ptr):
    for i in range(n):
        var bv_freq_squared = bv_freq_squared[unsafe_offset=i]
        if bv_freq_squared < 0.0:
            bv_freq_squared = nan[DType.float64]()
        dst[unsafe_offset=i] = sqrt(bv_freq_squared)


def brunt_vaisala_period(bv_freq_squared: Ptr, n: Int, dst: Ptr):
    for i in range(n):
        var bv_freq_squared = bv_freq_squared[unsafe_offset=i]
        if bv_freq_squared <= 0.0:
            bv_freq_squared = nan[DType.float64]()
        dst[unsafe_offset=i] = 2 * 3.141592653589793 / sqrt(bv_freq_squared)


def gradient_richardson_number(
    dthetadz: Ptr, dudz: Ptr, dvdz: Ptr, potential_temperature: Ptr, n: Int, dst: Ptr
):
    var i = 0
    while i + W <= n:
        var du = dudz.unsafe_load[width=W](i)
        var dv = dvdz.unsafe_load[width=W](i)
        dst.unsafe_store(
            i,
            F(g)
            / potential_temperature.unsafe_load[width=W](i)
            * (dthetadz.unsafe_load[width=W](i) / (du * du + dv * dv)),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = g / potential_temperature[unsafe_offset=i] * (
            dthetadz[unsafe_offset=i]
            / (dudz[unsafe_offset=i] ** 2 + dvdz[unsafe_offset=i] ** 2)
        )
        i += 1


def static_stability(
    pressure: Ptr, temperature: Ptr, dlogtheta_dp: Ptr, n: Int, dst: Ptr
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            (F(-Rd) * temperature.unsafe_load[width=W](i) / pressure.unsafe_load[width=W](i))
            * dlogtheta_dp.unsafe_load[width=W](i),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = (
            -Rd * temperature[unsafe_offset=i] / pressure[unsafe_offset=i]
        ) * dlogtheta_dp[unsafe_offset=i]
        i += 1


def vertical_velocity_pressure(
    w: Ptr, pressure: Ptr, temperature: Ptr, mixing_ratio: Ptr, n: Int, dst: Ptr
):
    var i = 0
    while i + W <= n:
        var rho = pressure.unsafe_load[width=W](i) / (
            F(Rd)
            * _virtual_temperature_v(
                temperature.unsafe_load[width=W](i),
                mixing_ratio.unsafe_load[width=W](i),
                epsilon,
            )
        )
        dst.unsafe_store(i, F(-g) * rho * w.unsafe_load[width=W](i))
        i += W
    while i < n:
        var rho = _density(
            pressure[unsafe_offset=i],
            temperature[unsafe_offset=i],
            mixing_ratio[unsafe_offset=i],
            epsilon,
        )
        dst[unsafe_offset=i] = -g * rho * w[unsafe_offset=i]
        i += 1


def _specific_humidity_from_dewpoint(pressure: Float64, dewpoint: Float64, phase: Int) -> Float64:
    var mixing_ratio = _saturation_mixing_ratio(pressure, dewpoint, phase)
    return mixing_ratio / (1 + mixing_ratio)


def _specific_humidity_from_dewpoint_chunk(
    pressure: Ptr, dewpoint: Ptr, phase: Int, lo: Int, hi: Int, dst: Ptr
):
    if phase != PHASE_LIQUID:
        for i in range(lo, hi):
            dst[unsafe_offset=i] = _specific_humidity_from_dewpoint(
                pressure[unsafe_offset=i], dewpoint[unsafe_offset=i], phase
            )
        return
    var i = lo
    while i + W <= hi:
        var w = _saturation_mixing_ratio_v(
            pressure.unsafe_load[width=W](i), dewpoint.unsafe_load[width=W](i)
        )
        dst.unsafe_store(i, w / (F(1.0) + w))
        i += W
    while i < hi:
        dst[unsafe_offset=i] = _specific_humidity_from_dewpoint(
            pressure[unsafe_offset=i], dewpoint[unsafe_offset=i], phase
        )
        i += 1


def specific_humidity_from_dewpoint(
    pressure: Ptr, dewpoint: Ptr, phase: Int, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _specific_humidity_from_dewpoint_chunk(pressure, dewpoint, phase, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _specific_humidity_from_dewpoint_chunk(
                pressure, dewpoint, phase, k * span, min((k + 1) * span, n), dst
            )
        parallelize(work, nw, nw)


def _dewpoint_from_specific_humidity_chunk(
    pressure: Ptr, specific_humidity: Ptr, lo: Int, hi: Int, dst: Ptr
):
    var i = lo
    while i + W <= hi:
        var q = specific_humidity.unsafe_load[width=W](i)
        var w = q / (F(1.0) - q)
        var e = pressure.unsafe_load[width=W](i) * w / (F(epsilon) + w)
        var val = log(e / F(sat_pressure_0c))
        dst.unsafe_store(i, F(zero_degc) + F(243.5) * val / (F(17.67) - val))
        i += W
    while i < hi:
        var q = specific_humidity[unsafe_offset=i]
        var w = q / (1 - q)
        var e = pressure[unsafe_offset=i] * w / (epsilon + w)
        dst[unsafe_offset=i] = _dewpoint(e)
        i += 1


def dewpoint_from_specific_humidity(
    pressure: Ptr, specific_humidity: Ptr, n: Int, dst: Ptr
):
    if n < PAR_MIN:
        _dewpoint_from_specific_humidity_chunk(pressure, specific_humidity, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _dewpoint_from_specific_humidity_chunk(
                pressure, specific_humidity, k * span, min((k + 1) * span, n), dst
            )
        parallelize(work, nw, nw)


def _moist_air_gas_constant(specific_humidity: Float64) -> Float64:
    return Rd + specific_humidity * (Rv - Rd)


def _moist_air_specific_heat_pressure(specific_humidity: Float64) -> Float64:
    return Cp_d + specific_humidity * (Cp_v - Cp_d)


def _relative_humidity_from_dewpoint(temperature: Float64, dewpoint: Float64, phase: Int) -> Float64:
    return _saturation_vapor_pressure(dewpoint, phase) / _saturation_vapor_pressure(
        temperature, phase
    )


# ---- scale height (thermo.py:5144) ----

def scale_height(temperature_bottom: Ptr, temperature_top: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        var t_bar = F(0.5) * (
            temperature_bottom.unsafe_load[width=W](i)
            + temperature_top.unsafe_load[width=W](i)
        )
        dst.unsafe_store(i, F(Rd) * t_bar / F(g))
        i += W
    while i < n:
        var t_bar = 0.5 * (
            temperature_bottom[unsafe_offset=i] + temperature_top[unsafe_offset=i]
        )
        dst[unsafe_offset=i] = Rd * t_bar / g
        i += 1
