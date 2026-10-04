"""Port of `metpy/calc/thermo.py` (MetPy 1.7.1), in upstream source order.

Signatures are upstream's.  The unit layer is not ported: every argument and
result is a plain float64 array in SI base units (kelvin, pascal, kg/kg), so
a `pint`/`xarray` quantity can be passed in and is unwrapped with
`.magnitude`, but nothing is converted on the way out.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._ffi import bcast, call, call2, call_scalar, call_masked, mag, phase_code
from .constants import epsilon
from .tools import first_derivative, get_layer


def moist_air_gas_constant(specific_humidity):
    (a,), shape = bcast(specific_humidity)
    return call("mm_moist_air_gas_constant", a, shape=shape)


def moist_air_specific_heat_pressure(specific_humidity):
    (a,), shape = bcast(specific_humidity)
    return call("mm_moist_air_specific_heat_pressure", a, shape=shape)


def moist_air_poisson_exponent(specific_humidity):
    (a,), shape = bcast(specific_humidity)
    return call("mm_moist_air_poisson_exponent", a, shape=shape)


def water_latent_heat_vaporization(temperature):
    (a,), shape = bcast(temperature)
    return call("mm_water_latent_heat_vaporization", a, shape=shape)


def water_latent_heat_sublimation(temperature):
    (a,), shape = bcast(temperature)
    return call("mm_water_latent_heat_sublimation", a, shape=shape)


def water_latent_heat_melting(temperature):
    (a,), shape = bcast(temperature)
    return call("mm_water_latent_heat_melting", a, shape=shape)


def relative_humidity_from_dewpoint(temperature, dewpoint, *, phase="liquid"):
    (t, td), shape = bcast(temperature, dewpoint)
    return call("mm_relative_humidity_from_dewpoint", t, td, phase_code(phase), shape=shape)


def exner_function(pressure, reference_pressure=100000.0):
    (p,), shape = bcast(pressure)
    return call("mm_exner_function", p, float(mag(reference_pressure)), shape=shape)


def potential_temperature(pressure, temperature):
    (p, t), shape = bcast(pressure, temperature)
    return call("mm_potential_temperature", p, t, shape=shape)


def temperature_from_potential_temperature(pressure, potential_temperature):
    (p, th), shape = bcast(pressure, potential_temperature)
    return call("mm_temperature_from_potential_temperature", p, th, shape=shape)


def dry_lapse(pressure, temperature, reference_pressure=None, vertical_dim=0):
    p, t = mag(pressure), mag(temperature)
    if reference_pressure is None:
        reference_pressure = p[(0,) * p.ndim]
    (pp, tt, ref), shape = bcast(p, t, np.broadcast_to(mag(reference_pressure), p.shape))
    return call("mm_dry_lapse", pp, tt, ref, shape=shape)


def lcl(pressure, temperature, dewpoint, max_iters=None, eps=None):
    (p, t, td), shape = bcast(pressure, temperature, dewpoint)
    return call2("mm_lcl", p, t, td, shape=shape)


def vapor_pressure(pressure, mixing_ratio):
    (p, w), shape = bcast(pressure, mixing_ratio)
    return call("mm_vapor_pressure", p, w, shape=shape)


def saturation_vapor_pressure(temperature, *, phase="liquid"):
    (t,), shape = bcast(temperature)
    return call("mm_saturation_vapor_pressure", t, phase_code(phase), shape=shape)


def dewpoint(vapor_pressure):
    (e,), shape = bcast(vapor_pressure)
    return call("mm_dewpoint", e, shape=shape)


def dewpoint_from_relative_humidity(temperature, relative_humidity):
    (t, rh), shape = bcast(temperature, relative_humidity)
    if np.any(rh > 1.2):
        warnings.warn("Relative humidity >120%, ensure proper units.", stacklevel=2)
    return call("mm_dewpoint_from_relative_humidity", t, rh, shape=shape)


def mixing_ratio(partial_press, total_press, molecular_weight_ratio=epsilon):
    (e, p), shape = bcast(partial_press, total_press)
    return call("mm_mixing_ratio", e, p, float(molecular_weight_ratio), shape=shape)


def saturation_mixing_ratio(total_press, temperature, *, phase="liquid"):
    (p, t), shape = bcast(total_press, temperature)
    return call("mm_saturation_mixing_ratio", p, t, phase_code(phase), shape=shape)


def equivalent_potential_temperature(pressure, temperature, dewpoint):
    (p, t, td), shape = bcast(pressure, temperature, dewpoint)
    return call("mm_equivalent_potential_temperature", p, t, td, shape=shape)


def virtual_temperature(temperature, mixing_ratio, molecular_weight_ratio=epsilon):
    (t, w), shape = bcast(temperature, mixing_ratio)
    return call("mm_virtual_temperature", t, w, float(molecular_weight_ratio), shape=shape)


def virtual_temperature_from_dewpoint(
    pressure, temperature, dewpoint, molecular_weight_ratio=epsilon, *, phase="liquid"
):
    (p, t, td), shape = bcast(pressure, temperature, dewpoint)
    return call(
        "mm_virtual_temperature_from_dewpoint", p, t, td,
        float(molecular_weight_ratio), phase_code(phase), shape=shape,
    )


def virtual_potential_temperature(
    pressure, temperature, mixing_ratio, molecular_weight_ratio=epsilon
):
    (p, t, w), shape = bcast(pressure, temperature, mixing_ratio)
    return call(
        "mm_virtual_potential_temperature", p, t, w,
        float(molecular_weight_ratio), shape=shape,
    )


def density(pressure, temperature, mixing_ratio, molecular_weight_ratio=epsilon):
    (p, t, w), shape = bcast(pressure, temperature, mixing_ratio)
    return call("mm_density", p, t, w, float(molecular_weight_ratio), shape=shape)


def psychrometric_vapor_pressure_wet(
    pressure, dry_bulb_temperature, wet_bulb_temperature, psychrometer_coefficient=0.000621
):
    (p, td, tw), shape = bcast(pressure, dry_bulb_temperature, wet_bulb_temperature)
    return call(
        "mm_psychrometric_vapor_pressure_wet", p, td, tw,
        float(psychrometer_coefficient), shape=shape,
    )


def relative_humidity_wet_psychrometric(
    pressure, dry_bulb_temperature, wet_bulb_temperature, **kwargs
):
    coefficient = float(kwargs.pop("psychrometer_coefficient", 0.000621))
    (p, td, tw), shape = bcast(pressure, dry_bulb_temperature, wet_bulb_temperature)
    return call(
        "mm_relative_humidity_wet_psychrometric", p, td, tw, coefficient, shape=shape,
    )


def mixing_ratio_from_relative_humidity(
    pressure, temperature, relative_humidity, *, phase="liquid"
):
    (p, t, rh), shape = bcast(pressure, temperature, relative_humidity)
    return call(
        "mm_mixing_ratio_from_relative_humidity", p, t, rh, phase_code(phase), shape=shape
    )


def relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio, *, phase="liquid"):
    (p, t, w), shape = bcast(pressure, temperature, mixing_ratio)
    return call(
        "mm_relative_humidity_from_mixing_ratio", p, t, w, phase_code(phase), shape=shape
    )


def mixing_ratio_from_specific_humidity(specific_humidity):
    (q,), shape = bcast(specific_humidity)
    return call("mm_mixing_ratio_from_specific_humidity", q, shape=shape)


def specific_humidity_from_mixing_ratio(mixing_ratio):
    (w,), shape = bcast(mixing_ratio)
    return call("mm_specific_humidity_from_mixing_ratio", w, shape=shape)


def relative_humidity_from_specific_humidity(
    pressure, temperature, specific_humidity, *, phase="liquid"
):
    (p, t, q), shape = bcast(pressure, temperature, specific_humidity)
    return call(
        "mm_relative_humidity_from_specific_humidity", p, t, q, phase_code(phase), shape=shape
    )


def dry_static_energy(height, temperature):
    (z, t), shape = bcast(height, temperature)
    return call("mm_dry_static_energy", z, t, shape=shape)


def moist_static_energy(height, temperature, specific_humidity):
    (z, t, q), shape = bcast(height, temperature, specific_humidity)
    return call("mm_moist_static_energy", z, t, q, shape=shape)


def _layer(p, t, name):
    p, t = mag(p), mag(t)
    if p.ndim > 1:
        raise NotImplementedError(
            f"{name}: upstream integrates along the last axis of an N-d profile; "
            "this port handles 1-D profiles"
        )
    return np.ascontiguousarray(p.reshape(-1)), np.ascontiguousarray(t.reshape(-1))


def thickness_hydrostatic(
    pressure, temperature, mixing_ratio=None, molecular_weight_ratio=epsilon, bottom=None,
    depth=None,
):
    # upstream's `bottom`/`depth` branch: the layer comes out of `get_layer`, which
    # returns it in descending pressure order, and the integral below is over
    # that order
    if bottom is not None or depth is not None:
        if mixing_ratio is None:
            p, t = get_layer(pressure, temperature, bottom=mag(bottom), depth=mag(depth))
        else:
            p, t, w = get_layer(pressure, temperature, mag(mixing_ratio), bottom=mag(bottom),
                                depth=mag(depth))
            t = virtual_temperature(t, w, molecular_weight_ratio)
        mixing_ratio = None
        pressure, temperature = p, t
    p, t = _layer(pressure, temperature, "thickness_hydrostatic")
    shape = (p.size,)
    w = np.zeros(shape, dtype=np.float64)
    if mixing_ratio is not None:
        w = np.ascontiguousarray(
            np.broadcast_to(mag(mixing_ratio), shape), dtype=np.float64
        )
    return call_scalar(
        "mm_thickness_hydrostatic", p, t, w, 0 if mixing_ratio is None else 1,
        float(molecular_weight_ratio), n=p.size,
    )


def thickness_hydrostatic_from_relative_humidity(
    pressure, temperature, relative_humidity, bottom=None, depth=None
):
    mixing = mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity)
    return thickness_hydrostatic(pressure, temperature, mixing_ratio=mixing, bottom=bottom,
                                 depth=depth)


def brunt_vaisala_frequency_squared(height, potential_temperature, vertical_dim=0):
    z, th = mag(height), mag(potential_temperature)
    dtheta_dz = first_derivative(th, axis=vertical_dim, x=z)
    (a, b, c), shape = bcast(z, dtheta_dz, th)
    return call("mm_brunt_vaisala_frequency_squared", a, b, c, shape=shape)


def brunt_vaisala_frequency(height, potential_temperature, vertical_dim=0):
    squared = brunt_vaisala_frequency_squared(
        height, potential_temperature, vertical_dim=vertical_dim
    )
    return call("mm_brunt_vaisala_frequency", squared, shape=squared.shape)


def brunt_vaisala_period(height, potential_temperature, vertical_dim=0):
    squared = brunt_vaisala_frequency_squared(
        height, potential_temperature, vertical_dim=vertical_dim
    )
    return call("mm_brunt_vaisala_period", squared, shape=squared.shape)


def gradient_richardson_number(height, potential_temperature, u, v, vertical_dim=0):
    z, th = mag(height), mag(potential_temperature)
    dthetadz = first_derivative(th, axis=vertical_dim, x=z)
    dudz = first_derivative(mag(u), axis=vertical_dim, x=np.broadcast_to(z, np.shape(u)))
    dvdz = first_derivative(mag(v), axis=vertical_dim, x=np.broadcast_to(z, np.shape(v)))
    (a, b, c, d), shape = bcast(dthetadz, dudz, dvdz, th)
    return call("mm_gradient_richardson_number", a, b, c, d, shape=shape)


def static_stability(pressure, temperature, vertical_dim=0):
    p, t = mag(pressure), mag(temperature)
    theta = potential_temperature(p, t)
    dlogtheta_dp = first_derivative(np.log(theta), axis=vertical_dim, x=p)
    (a, b, c), shape = bcast(p, t, dlogtheta_dp)
    return call("mm_static_stability", a, b, c, shape=shape)


def vertical_velocity_pressure(w, pressure, temperature, mixing_ratio=0):
    (a, b, c, d), shape = bcast(w, pressure, temperature, mixing_ratio)
    return call("mm_vertical_velocity_pressure", a, b, c, d, shape=shape)


def dewpoint_from_specific_humidity(pressure, specific_humidity):
    (p, q), shape = bcast(pressure, specific_humidity)
    return call("mm_dewpoint_from_specific_humidity", p, q, shape=shape)


def specific_humidity_from_dewpoint(pressure, dewpoint, *, phase="liquid"):
    (p, td), shape = bcast(pressure, dewpoint)
    return call("mm_specific_humidity_from_dewpoint", p, td, phase_code(phase), shape=shape)


def scale_height(temperature_bottom, temperature_top):
    (a, b), shape = bcast(temperature_bottom, temperature_top)
    return call("mm_scale_height", a, b, shape=shape)


def mixed_layer(pressure, *args, height=None, bottom=None, depth=None, interpolate=True):
    """Upstream `thermo.mixed_layer`; `bottom` and `depth` are pressures in Pa."""
    layer = get_layer(pressure, *args, height=height, bottom=bottom, depth=depth,
                      interpolate=interpolate)
    p_layer = layer[0]
    datavars_layer = layer[1:]

    ret = []
    for datavar_layer in datavars_layer:
        actual_depth = abs(p_layer[0] - p_layer[-1])
        ret.append(np.trapezoid(datavar_layer, p_layer) / -actual_depth)
    return ret
