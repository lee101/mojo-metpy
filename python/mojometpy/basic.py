"""Port of `metpy/calc/basic.py` (MetPy 1.7.1), in upstream source order.

Signatures are upstream's.  The unit layer is not ported, so the three
"apparent temperature" functions take the units upstream converts to
internally: degrees Celsius for `windchill`, degrees Fahrenheit for
`heat_index`, and kilometres per hour for the wind speed.  The piecewise
definitions keep upstream's branch order, and the masked regions are returned
as `numpy.ma.MaskedArray` exactly when upstream would return one.
"""

from __future__ import annotations

import numpy as np

from . import _lib
from ._ffi import bcast, call, call2, call_masked, mag, with_mask


def _check_radians(value, max_radians):
    value = mag(value)
    if np.any(value > max_radians):
        raise ValueError(f"Value must be in radians, less than {max_radians}.")
    return value


def wind_speed(u, v):
    (a, b), shape = bcast(u, v)
    return call("mm_wind_speed", a, b, shape=shape)


def wind_direction(u, v, convention="from"):
    if convention not in ("to", "from"):
        raise ValueError('Invalid kwarg for "convention". Valid options are "from" or "to".')
    (a, b), shape = bcast(u, v)
    return call("mm_wind_direction", a, b, 1 if convention == "to" else 0, shape=shape)


def wind_components(speed, wind_direction):
    direction = _check_radians(wind_direction, 4 * np.pi)
    (s, d), shape = bcast(speed, direction)
    return call2("mm_wind_components", s, d, shape=shape)


def windchill(temperature, speed, face_level_winds=False, mask_undefined=True):
    (t, s), shape = bcast(temperature, speed)
    values, mask = call_masked(
        "mm_windchill", t, s, face_level_winds, mask_undefined, shape=shape
    )
    return with_mask(values, mask)


def heat_index(temperature, relative_humidity, mask_undefined=True):
    (t, rh), shape = bcast(temperature, relative_humidity)
    values, mask = call_masked("mm_heat_index", t, rh, mask_undefined, shape=shape)
    return with_mask(values, mask)


def apparent_temperature(
    temperature, relative_humidity, speed, face_level_winds=False, mask_undefined=True
):
    (t, rh, s), shape = bcast(temperature, relative_humidity, speed)
    values, mask = call_masked(
        "mm_apparent_temperature", t, rh, s, face_level_winds, mask_undefined, shape=shape
    )
    return with_mask(values, mask)


def pressure_to_height_std(pressure):
    (p,), shape = bcast(pressure)
    return call("mm_pressure_to_height_std", p, shape=shape)


def height_to_geopotential(height):
    (z,), shape = bcast(height)
    return call("mm_height_to_geopotential", z, shape=shape)


def geopotential_to_height(geopotential):
    (z,), shape = bcast(geopotential)
    return call("mm_geopotential_to_height", z, shape=shape)


def height_to_pressure_std(height):
    (z,), shape = bcast(height)
    return call("mm_height_to_pressure_std", z, shape=shape)


def coriolis_parameter(latitude):
    lat = _check_radians(latitude, np.pi / 2)
    (a,), shape = bcast(lat)
    return call("mm_coriolis_parameter", a, shape=shape)


def add_height_to_pressure(pressure, height):
    (p, z), shape = bcast(pressure, height)
    return call("mm_add_height_to_pressure", p, z, shape=shape)


def add_pressure_to_height(height, pressure):
    (z, p), shape = bcast(height, pressure)
    return call("mm_add_pressure_to_height", z, p, shape=shape)


def sigma_to_pressure(sigma, pressure_sfc, pressure_top):
    sig, sfc, top = mag(sigma), mag(pressure_sfc), mag(pressure_top)
    if np.any(sig < 0) or np.any(sig > 1):
        raise ValueError("Sigma values should be bounded by 0 and 1")
    if np.any(sfc < 0) or np.any(top < 0):
        raise ValueError("Pressure input should be non-negative")
    (a, b, c), shape = bcast(sig, sfc, top)
    return call("mm_sigma_to_pressure", a, b, c, shape=shape)


def altimeter_to_station_pressure(altimeter_value, height):
    (a, z), shape = bcast(altimeter_value, height)
    return call("mm_altimeter_to_station_pressure", a, z, shape=shape)


def altimeter_to_sea_level_pressure(altimeter_value, height, temperature):
    (a, z, t), shape = bcast(altimeter_value, height, temperature)
    return call("mm_altimeter_to_sea_level_pressure", a, z, t, shape=shape)


def smooth_window(scalar_grid, window, passes=1, normalize_weights=True):
    """Upstream `basic.smooth_window`, restricted to 2-D grids."""
    data = np.array(mag(scalar_grid), dtype=np.float64, order="C", copy=True)
    w = mag(window)
    if data.ndim != 2:
        raise NotImplementedError(
            "smooth_window: this port handles 2-D grids; upstream also accepts N-d"
        )
    if w.ndim == 1:
        w = w.reshape(1, -1)
    if w.ndim != 2 or any(size % 2 == 0 for size in w.shape):
        raise ValueError("The shape of the smoothing window must be odd in all dimensions.")
    weights = np.ascontiguousarray(w.reshape(-1), dtype=np.float64)
    if normalize_weights:
        _lib.lib().mm_normalize_window(weights.ctypes.data, weights.size)
    scratch = np.empty_like(data)
    _lib.lib().mm_smooth_window(
        data.ctypes.data, scratch.ctypes.data, weights.ctypes.data, w.shape[0], w.shape[1],
        data.shape[0], data.shape[1], int(passes),
    )
    return scratch


def smooth_n_point(scalar_grid, n=5, passes=1):
    if n == 9:
        weights = np.array(
            [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
        )
    elif n == 5:
        weights = np.array(
            [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]]
        )
    else:
        raise ValueError(
            "The number of points to use in the smoothing calculation must be either 5 or 9."
        )
    return smooth_window(scalar_grid, window=weights, passes=passes, normalize_weights=False)


def smooth_rectangular(scalar_grid, size, passes=1):
    return smooth_window(scalar_grid, np.ones(size), passes=passes)


def smooth_circular(scalar_grid, radius, passes=1):
    size = 2 * radius + 1
    x, y = np.mgrid[:size, :size]
    distance = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
    circle = distance <= radius
    return smooth_window(scalar_grid, circle, passes=passes)
