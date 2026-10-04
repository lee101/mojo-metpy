"""Port of `metpy/calc/indices.py` (MetPy 1.7.1), in upstream source order.

`precipitable_water`, `mean_pressure_weighted`,
`weighted_continuous_average`, `bunkers_storm_motion`,
`corfidi_storm_motion`, `bulk_shear` and `critical_angle` all route through
`get_layer`, which is ported here, so they are too.

Units: the argument conventions are in the README's unit table.  The two places
upstream pins an output unit with `.to(...)` are kept: `precipitable_water`
returns millimetres (numerically identical to the kg/m**2 it integrates) and
`critical_angle` returns degrees.

Upstream picks the length branch of `get_layer` from the units of `bottom` and
`depth`.  Here the public `get_layer` is the pressure branch and
`get_layer_by_length` is the length branch; the callers below that upstream
always passes metres to go through the latter.
"""

from __future__ import annotations

import numpy as np

from ._ffi import bcast, call, mag
from .basic import wind_speed
from .constants import g as _G, rho_l as _rho_l
from .thermo import mixing_ratio, saturation_vapor_pressure
from .tools import get_layer, get_layer_by_length as _get_layer_by_length


def _remove_nans(*variables):
    """Upstream `tools._remove_nans`: drop every level with a missing value."""
    arrays = [np.atleast_1d(mag(v)).astype(np.float64) for v in variables]
    good = ~np.logical_or.reduce([np.isnan(a) for a in arrays])
    return tuple(a[good] for a in arrays)


def precipitable_water(pressure, dewpoint, *, bottom=None, top=None):
    """Upstream `indices.precipitable_water`, in millimetres.

    `dewpoint` is in K; upstream takes degrees Celsius and this port's dewpoint
    convention applies here too.
    """
    pressure, dewpoint = _remove_nans(pressure, dewpoint)

    min_pressure = np.nanmin(pressure)
    max_pressure = np.nanmax(pressure)

    if top is None:
        top = min_pressure
    elif not min_pressure <= top <= max_pressure:
        raise ValueError(f"The pressure and dewpoint profile ranges from {max_pressure} to "
                         f"{min_pressure}, after removing missing values. {top} is outside "
                         "this range.")

    if bottom is None:
        bottom = max_pressure
    elif not min_pressure <= bottom <= max_pressure:
        raise ValueError(f"The pressure and dewpoint profile ranges from {max_pressure} to "
                         f"{min_pressure}, after removing missing values. {bottom} is outside "
                         "this range.")

    pres_layer, dewpoint_layer = get_layer(pressure, dewpoint, bottom=bottom,
                                           depth=bottom - top)

    w = mixing_ratio(saturation_vapor_pressure(dewpoint_layer), pres_layer)

    # Since pressure is in decreasing order, pw will be the opposite sign of
    # that expected.
    return -np.trapezoid(w, pres_layer) / (_G * _rho_l)


def mean_pressure_weighted(pressure, *args, height=None, bottom=None, depth=None):
    """Upstream `indices.mean_pressure_weighted`; `bottom` and `depth` in Pa."""
    pres_prof, *others = get_layer(pressure, *args, height=height, bottom=bottom,
                                   depth=depth)

    pres_int = 0.5 * (pres_prof[-1] ** 2 - pres_prof[0] ** 2)

    return [np.trapezoid(var_prof * pres_prof, x=pres_prof) / pres_int
            for var_prof in others]


def weighted_continuous_average(pressure, *args, height=None, bottom=None, depth=None):
    """Upstream `indices.weighted_continuous_average`; `bottom`/`depth` in Pa."""
    pres_prof, *others = get_layer(pressure, *args, height=height, bottom=bottom,
                                   depth=depth)

    return [np.trapezoid(var_prof, x=pres_prof) / (pres_prof[-1] - pres_prof[0])
            for var_prof in others]


def _weighted_continuous_average_by_length(pressure, u, v, *, height, depth,
                                            bottom=None):
    """`weighted_continuous_average` over a layer whose depth is in metres."""
    pres_prof, u_prof, v_prof = _get_layer_by_length(
        pressure, u, v, height=height, bottom=bottom, depth=depth
    )
    span = pres_prof[-1] - pres_prof[0]
    return (np.trapezoid(u_prof, x=pres_prof) / span,
            np.trapezoid(v_prof, x=pres_prof) / span)


def bunkers_storm_motion(pressure, u, v, height):
    """Upstream `indices.bunkers_storm_motion`; `height` is in m."""
    pressure, u, v, height = _remove_nans(pressure, u, v, height)

    def layer_mean(depth, bottom=None):
        u_mean, v_mean = _weighted_continuous_average_by_length(
            pressure, u, v, height=height, depth=depth, bottom=bottom
        )
        return np.array([u_mean, v_mean])

    # mean wind from sfc-6km
    wind_mean = layer_mean(6000.0)
    # mean wind from sfc-500m
    wind_500m = layer_mean(500.0)
    # mean wind from 5.5-6km
    wind_5500m = layer_mean(500.0, bottom=height[0] + 5500.0)

    # Calculate the shear vector from sfc-500m to 5.5-6km
    shear = wind_5500m - wind_500m

    # Take the cross product of the wind shear and k, and divide by the vector
    # magnitude and multiply by the deviation empirically calculated in Bunkers
    # (2000) (7.5 m/s)
    shear_cross = np.array([shear[1], -shear[0]])
    shear_mag = np.hypot(*shear)
    rdev = shear_cross * (7.5 / shear_mag)

    # Add the deviations to the layer average wind to get the RM motion
    right_mover = wind_mean + rdev
    # Subtract the deviations to get the LM motion
    left_mover = wind_mean - rdev
    return right_mover, left_mover, wind_mean


def corfidi_storm_motion(pressure, u, v, *, u_llj=None, v_llj=None):
    """Upstream `indices.corfidi_storm_motion`; pressures in Pa."""
    if (u_llj is None) ^ (v_llj is None):
        raise ValueError("Must specify both u_llj and v_llj or neither")

    pressure, u, v = _remove_nans(pressure, u, v)

    if u_llj is not None and v_llj is not None:
        llj_inverse = np.array([-mag(u_llj), -mag(v_llj)])
    elif np.max(pressure) >= 85000.0:
        wind_magnitude = wind_speed(u, v)
        lowlevel_index = np.argmin(pressure >= 85000.0)
        llj_index = np.argmax(wind_magnitude[:lowlevel_index])
        llj_inverse = np.array([-u[llj_index], -v[llj_index]])
    else:
        raise ValueError("Must specify low-level jet or "
                         "specify pressure values below 850 hPa")

    # cloud layer mean wind; don't select outside bounds of given data
    if pressure[0] < 85000.0:
        bottom = pressure[0]
    else:
        bottom = 85000.0
    if pressure[-1] > 30000.0:
        depth = bottom - pressure[-1]
    else:
        depth = 55000.0
    cloud_layer_winds = np.array(mean_pressure_weighted(pressure, u, v, bottom=bottom,
                                                        depth=depth))

    upwind = cloud_layer_winds + llj_inverse
    downwind = 2 * cloud_layer_winds + llj_inverse
    return upwind, downwind


def _bulk_shear(pressure, u, v, height, bottom, depth, by_length):
    """Upstream `bulk_shear`, with its depth branch chosen explicitly."""
    get = _get_layer_by_length if by_length else get_layer
    _, u_layer, v_layer = get(pressure, u, v, height=height, bottom=bottom, depth=depth)
    return u_layer[-1] - u_layer[0], v_layer[-1] - v_layer[0]


def bulk_shear(pressure, u, v, height=None, bottom=None, depth=None):
    """Upstream `indices.bulk_shear`; `bottom` and `depth` are pressures in Pa."""
    return _bulk_shear(pressure, u, v, height, bottom, depth, False)


def supercell_composite(mucape, effective_storm_helicity, effective_shear):
    (a, b, c), shape = bcast(mucape, effective_storm_helicity, effective_shear)
    return call("mm_supercell_composite", a, b, c, shape=shape)


def significant_tornado(sbcape, surface_based_lcl_height, storm_helicity_1km, shear_6km):
    arrays, shape = bcast(sbcape, surface_based_lcl_height, storm_helicity_1km, shear_6km)
    return call("mm_significant_tornado", *arrays, shape=shape)


def critical_angle(pressure, u, v, height, u_storm, v_storm):
    """Upstream `indices.critical_angle`, in degrees."""
    pressure, u, v, height = (mag(a) for a in (pressure, u, v, height))
    u_storm, v_storm = mag(u_storm), mag(v_storm)

    sort_inds = np.argsort(pressure[::-1])
    pressure = pressure[sort_inds]
    height = height[sort_inds]
    u = u[sort_inds]
    v = v[sort_inds]

    # Calculate sfc-500m shear vector
    shr5 = _bulk_shear(pressure, u, v, height, None, 500.0, True)
    # Make everything relative to the sfc wind orientation
    umn = u_storm - u[0]
    vmn = v_storm - v[0]
    vshr = np.array([shr5[0], shr5[1]])
    vsm = np.array([umn, vmn])
    angle_c = np.dot(vshr, vsm) / (np.linalg.norm(vshr) * np.linalg.norm(vsm))
    return np.degrees(np.arccos(angle_c))
