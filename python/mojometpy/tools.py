"""Port of `metpy/calc/tools.py` and `metpy/interpolate/one_dimension.py`.

Names, argument order and defaults follow upstream.  The derivative helpers
materialise `delta` as a length-`n-1` vector before it crosses into the
kernel, so the kernel never has to know about broadcasting.

Units are gone, so the two places upstream dispatches on the unit of a bound
have one answer each: `get_layer` treats `bottom` and `depth` as pressures in
Pa (its own default depth is 100 hPa) and `get_layer_heights` treats them as
lengths in m.  See the README's coverage section.
"""

from __future__ import annotations

import numpy as np

from . import _lib
from ._ffi import _arg, axis_geometry, call, call_axis, call_int, mag, size


# ---- interpolation (metpy/interpolate/one_dimension.py:52) ----


def _interp(x, xp, values, fill_value, out_shape):
    """The kernel pass shared by `interpolate_1d` and `log_interpolate_1d`.

    `xp` and `values` are both line-last here, so one `(n, 1, outer)` geometry
    describes every line of every axis.
    """
    n, stride, outer = axis_geometry(xp.shape, -1)
    out = np.empty(size(out_shape), dtype=np.float64)
    _lib.lib().mm_interpolate_1d(
        _arg(x), _arg(xp), _arg(values), x.size, n, stride, outer,
        float(fill_value), out.ctypes.data,
    )
    return out.reshape(out_shape)


def interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan, return_list_always=False):
    """Upstream `metpy.interpolate.interpolate_1d`, linear in `x`."""
    x = np.ascontiguousarray(mag(x).reshape(-1))
    xp = mag(xp)
    variables = [mag(a) for a in args]
    if xp.ndim == 0 or not variables:
        raise ValueError("xp and args must have at least one dimension, and args "
                         "must not be empty.")
    if any(v.shape != xp.shape for v in variables):
        raise ValueError("xp and args must be the same shape.")

    final_shape = np.broadcast_shapes(xp.shape, *(v.shape for v in variables))
    axis_n = axis + len(final_shape) if axis < 0 else axis

    order = np.argsort(xp, axis=axis_n)
    xp_sorted = np.ascontiguousarray(
        np.moveaxis(np.take_along_axis(np.broadcast_to(xp, final_shape), order,
                                       axis=axis_n), axis_n, -1)
    )

    x_sorted = np.ascontiguousarray(x[np.argsort(x)])
    descending = bool(x.size) and x[0] > x[-1]
    out_shape = final_shape[:axis_n] + (x.size,) + final_shape[axis_n + 1:]

    ret = []
    for v in variables:
        moved = np.ascontiguousarray(
            np.moveaxis(np.take_along_axis(np.broadcast_to(v, final_shape), order,
                                           axis=axis_n), axis_n, -1)
        )
        line_shape = moved.shape[:-1] + (x_sorted.size,)
        interp = _interp(x_sorted, xp_sorted, moved, fill_value, line_shape)
        if descending:
            interp = interp[..., ::-1]
        ret.append(np.ascontiguousarray(np.moveaxis(interp, -1, axis_n)))

    if return_list_always or len(ret) > 1:
        return ret
    return ret[0]


def log_interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan):
    """Upstream `metpy.interpolate.log_interpolate_1d`: log in `x`, then linear."""
    return interpolate_1d(
        np.log(mag(x)), np.log(mag(xp)), *args, axis=axis, fill_value=fill_value
    )


# ---- tools.py:52 ----


def resample_nn_1d(a, centers):
    """Upstream `tools.resample_nn_1d`: nearest-neighbour indexes, deduplicated."""
    a = np.ascontiguousarray(mag(a).reshape(-1))
    ix = []
    for center in mag(centers).reshape(-1):
        index = call_int("mm_resample_nn_1d_nearest", a, float(center), a.size)
        if index not in ix:
            ix.append(index)
    return ix


# ---- tools.py:78 ----


def nearest_intersection_idx(a, b):
    """Upstream `tools.nearest_intersection_idx`."""
    a = np.ascontiguousarray(mag(a).reshape(-1))
    b = np.ascontiguousarray(mag(b).reshape(-1))
    out = np.empty(a.size, dtype=np.float64)
    found = call_int("mm_nearest_intersection_idx", a, b, a.size, out.ctypes.data)
    return out[:found].astype(int)


# ---- tools.py:328 ----


def _greater_or_close(a, value):
    return (a > value) | np.isclose(a, value)


def _less_or_close(a, value):
    return (a < value) | np.isclose(a, value)


def _get_bound_pressure_height(pressure, bound, height=None, interpolate=True,
                              bound_is_length=False):
    """Upstream `_get_bound_pressure_height`.

    `bound_is_length` selects the branch upstream picks with `bound.check(...)`,
    which is the one thing the missing unit layer cannot decide at runtime.
    """
    from .basic import height_to_pressure_std, pressure_to_height_std

    pressure = mag(pressure)
    bound = mag(bound)
    if height is not None:
        height = mag(height)
    sort_inds = np.argsort(pressure)[::-1]
    pressure = pressure[sort_inds]
    if height is not None:
        height = height[sort_inds]

    if bound_is_length:
        if height is not None:
            if np.any(height == bound):
                bound_height = np.float64(bound)
                bound_pressure = pressure[height == bound]
            elif interpolate:
                bound_height = np.float64(bound)
                bound_pressure = np.interp(
                    np.atleast_1d(np.float64(bound)), height, pressure
                )
            else:
                idx = (np.abs(height - bound)).argmin()
                bound_pressure = pressure[idx]
                bound_height = height[idx]
        else:
            bound_height = np.float64(bound)
            bound_pressure = height_to_pressure_std(bound)
            if not interpolate:
                idx = (np.abs(pressure - bound_pressure)).argmin()
                bound_pressure = pressure[idx]
                bound_height = pressure_to_height_std(bound_pressure)
    elif np.any(pressure == bound):
        bound_pressure = np.atleast_1d(np.float64(bound))
        if height is not None:
            bound_height = height[pressure == bound]
        else:
            bound_height = pressure_to_height_std(bound_pressure)
    elif interpolate:
        bound_pressure = np.float64(bound)
        if height is not None:
            bound_height = log_interpolate_1d(bound_pressure, pressure, height)
        else:
            bound_height = pressure_to_height_std(bound_pressure)
    else:
        idx = (np.abs(pressure - bound)).argmin()
        bound_pressure = pressure[idx]
        if height is not None:
            bound_height = height[idx]
        else:
            bound_height = pressure_to_height_std(bound_pressure)

    if not (_greater_or_close(bound_pressure, np.nanmin(pressure))
            and _less_or_close(bound_pressure, np.nanmax(pressure))):
        raise ValueError("Specified bound is outside pressure range.")
    if height is not None and not (_less_or_close(bound_height, np.nanmax(height))
                                   and _greater_or_close(bound_height, np.nanmin(height))):
        raise ValueError("Specified bound is outside height range.")

    return bound_pressure, bound_height


# ---- tools.py:440 ----


def get_layer_heights(height, depth, *args, bottom=None, interpolate=True, with_agl=False):
    """Upstream `tools.get_layer_heights`; `bottom` and `depth` are metres."""
    height = mag(height)
    for datavar in args:
        if len(height) != len(mag(datavar)):
            raise ValueError("Height and data variables must have the same length.")

    if with_agl:
        sfc_height = np.min(height)
        height = height - sfc_height

    if bottom is None:
        bottom = height[0]
    bottom = mag(bottom)
    depth = mag(depth)
    top = bottom + depth

    sort_inds = np.argsort(height)
    height = height[sort_inds]

    inds = _greater_or_close(height, bottom) & _less_or_close(height, top)
    heights_interp = height[inds]

    if interpolate:
        if not np.any(heights_interp == top):
            heights_interp = np.sort(np.append(heights_interp, top))
        if not np.any(heights_interp == bottom):
            heights_interp = np.sort(np.append(heights_interp, bottom))

    ret = [np.ascontiguousarray(heights_interp)]

    for datavar in args:
        datavar = mag(datavar)[sort_inds]
        if interpolate:
            datavar = np.ascontiguousarray(
                interpolate_1d(np.ascontiguousarray(heights_interp),
                               np.ascontiguousarray(height), datavar)
            )
        else:
            datavar = datavar[inds]
        ret.append(np.ascontiguousarray(datavar))
    return ret


# ---- tools.py:538 ----


def _get_layer(pressure, args, height, bottom, depth, interpolate, bound_is_length,
               depth_is_length):
    """Upstream `get_layer` with the unit branches selected explicitly."""
    pressure = mag(pressure)
    if depth is None:
        depth = 10000.0
    depth = mag(depth)

    for datavar in args:
        if len(pressure) != len(mag(datavar)):
            raise ValueError("Pressure and data variables must have the same length.")
    # upstream's default `bottom = np.nanmax(pressure)` carries pressure units,
    # so the length branch only applies to a bound the caller actually passed
    bottom_is_length = bound_is_length and bottom is not None
    if bottom is None:
        bottom = np.nanmax(pressure)
    bottom = mag(bottom)
    bottom_pressure, bottom_height = _get_bound_pressure_height(
        pressure, bottom, height=height, interpolate=interpolate,
        bound_is_length=bottom_is_length,
    )

    if depth_is_length:
        top = bottom_height + depth
    else:
        top = bottom_pressure - depth

    top_pressure, _ = _get_bound_pressure_height(
        pressure, top, height=height, interpolate=interpolate, bound_is_length=depth_is_length
    )

    sort_inds = np.argsort(pressure)
    pressure = pressure[sort_inds]

    inds = (_less_or_close(pressure, bottom_pressure)
            & _greater_or_close(pressure, top_pressure))
    p_interp = pressure[inds]

    if interpolate:
        if not np.any(np.isclose(top_pressure, p_interp)):
            p_interp = np.sort(np.append(p_interp, top_pressure))
        if not np.any(np.isclose(bottom_pressure, p_interp)):
            p_interp = np.sort(np.append(p_interp, bottom_pressure))

    ret = [np.ascontiguousarray(p_interp[::-1])]

    for datavar in args:
        datavar = mag(datavar)[sort_inds]
        if interpolate:
            datavar = np.ascontiguousarray(
                log_interpolate_1d(np.ascontiguousarray(p_interp),
                                   np.ascontiguousarray(pressure), datavar)
            )
        else:
            datavar = datavar[inds]
        ret.append(np.ascontiguousarray(datavar[::-1]))
    return ret


def get_layer(pressure, *args, height=None, bottom=None, depth=None, interpolate=True):
    """Upstream `tools.get_layer`; `bottom` and `depth` are pressures in Pa."""
    return _get_layer(pressure, args, height, bottom, depth, interpolate, False, False)


def get_layer_by_length(pressure, *args, height=None, bottom=None, depth=None,
                        interpolate=True):
    """`get_layer` with `bottom` and `depth` as lengths in m.

    Upstream chooses this branch from the units of the arguments; several
    upstream callers (`bunkers_storm_motion`, `critical_angle`, ...) always pass
    metres, and they come through here.
    """
    return _get_layer(pressure, args, height, bottom, depth, interpolate, True, True)


# ---- tools.py:789 ----


def lat_lon_grid_deltas(longitude, latitude, x_dim=-1, y_dim=-2):
    """Upstream `tools.lat_lon_grid_deltas` for the default spherical `geod`.

    Upstream returns `pint` Quantities; this returns the same distances in m.
    """
    longitude = mag(longitude)
    latitude = mag(latitude)
    if latitude.ndim != longitude.ndim:
        raise ValueError("Latitude and longitude must have the same number of dimensions.")

    if latitude.ndim < 2:
        longitude, latitude = np.meshgrid(longitude, latitude)
    longitude = np.ascontiguousarray(longitude, dtype=np.float64)
    latitude = np.ascontiguousarray(latitude, dtype=np.float64)

    dx = call_axis("mm_lat_lon_grid_deltas_dx", longitude, latitude,
                   shape=longitude.shape, axis=x_dim)
    dy = call_axis("mm_lat_lon_grid_deltas_dy", longitude, latitude,
                   shape=longitude.shape, axis=y_dim)
    return dx, dy


# ---- tools.py:1197 ----



def _broadcast_to_axis(arr, axis, ndim):
    """Upstream's `_broadcast_to_axis`: put the values along the given axis."""
    if arr.ndim == 1 and arr.ndim < ndim:
        new_shape = [1] * ndim
        new_shape[axis] = arr.size
        arr = arr.reshape(*new_shape)
    return arr


def _positions_to_spacing(positions, axis, expected, is_x):
    """Collapse upstream's broadcast `x`/`delta` down the axis it spaces.

    Upstream lets numpy broadcast the positions against `f` and then slices
    both with `make_take`, so a full grid of spacings, a length-`n` coordinate
    vector and a scalar all describe the same derivative.  The kernel here
    reads the spacing as a length-`n-1` vector shared by every line, which is
    exactly upstream's `_broadcast_to_axis` contract, so a grid is collapsed to
    its first line and a coordinate vector is differenced.

    `expected` is the length the caller must supply along `axis`: `n` for `x`
    and `n-1` for `delta`.
    """
    positions = np.asarray(mag(positions))
    if positions.ndim == 0:
        return np.full(expected, float(positions), dtype=np.float64)
    if positions.ndim == 1:
        if positions.size != expected:
            raise ValueError(
                'The size of "x" or "delta" must match the size of "f" along "axis".'
            )
        if is_x:
            return np.ascontiguousarray(np.diff(positions), dtype=np.float64)
        return np.ascontiguousarray(positions, dtype=np.float64)
    moved = np.moveaxis(positions, axis, 0)
    first = moved[0] if moved.ndim > 1 else moved
    try:
        line = np.broadcast_to(first, (expected,))
    except ValueError:
        raise ValueError(
            'The shape of "x" or "delta" must be broadcastable against the shape '
            'of "f" along "axis".'
        ) from None
    if is_x:
        line = np.diff(line)
    return np.ascontiguousarray(line, dtype=np.float64)


def _process_deriv_args(f, axis, x, delta):
    """Upstream's `_process_deriv_args`, minus the pint/xarray branches.

    Returns `(f, axis, delta)` where `delta` is the length-`n-1` spacing vector
    the kernels index; upstream keeps a broadcast array here instead, which is
    the only structural difference between the two.
    """
    f = mag(f)
    n = f.ndim
    if n == 0:
        raise ValueError("f must have at least 3 point along the desired axis.")
    axis = 0 if axis is None else int(axis)
    axis = axis + n if axis < 0 else axis
    if f.shape[axis] < 3:
        raise ValueError("f must have at least 3 point along the desired axis.")
    length = f.shape[axis]
    if delta is not None:
        if x is not None:
            raise ValueError('Cannot specify both "x" and "delta".')
        delta = _positions_to_spacing(delta, axis, length - 1, is_x=False)
    elif x is not None:
        delta = _positions_to_spacing(x, axis, length, is_x=True)
    else:
        raise ValueError('Must specify either "x" or "delta" for value positions.')
    return f, axis, delta


def _deriv(fn, f, axis, x, delta):
    f, axis, delta = _process_deriv_args(f, axis, x, delta)
    shape = f.shape
    f = np.ascontiguousarray(f, dtype=np.float64)
    n, stride, outer = axis_geometry(shape, axis)
    return call(fn, f, delta, n, stride, outer, shape=shape)


def first_derivative(f, axis=None, x=None, delta=None):
    """Centered first derivative along `axis` (upstream `tools.first_derivative`)."""
    return _deriv("mm_first_derivative", f, axis, x, delta)


def second_derivative(f, axis=None, x=None, delta=None):
    """Centered second derivative along `axis` (upstream `tools.second_derivative`)."""
    return _deriv("mm_second_derivative", f, axis, x, delta)


def _process_gradient_args(f, axes, coordinates, deltas):
    f = mag(f)
    axes_given = axes is not None
    axes = list(axes) if axes_given else list(range(f.ndim))
    axes = [a + f.ndim if a < 0 else a for a in axes]

    def _check_length(positions):
        if axes_given and len(positions) < len(axes):
            raise ValueError(
                'Length of "coordinates" or "deltas" cannot be less than that of "axes".'
            )
        if not axes_given and len(positions) != len(axes):
            raise ValueError(
                'Length of "coordinates" or "deltas" must match the number of '
                "dimensions of `f` when `axes` is not given."
            )

    if deltas is not None:
        if coordinates is not None:
            raise ValueError('Cannot specify both "coordinates" and "deltas".')
        _check_length(deltas)
        return "delta", deltas, axes
    if coordinates is not None:
        _check_length(coordinates)
        return "x", coordinates, axes
    raise ValueError(
        'Must specify either "coordinates" or "deltas" for value positions '
        "when `f` is not a DataArray."
    )


def gradient(f, axes=None, coordinates=None, deltas=None):
    """Upstream `tools.gradient`: one `first_derivative` per requested axis."""
    f = mag(f)
    kwarg, positions, axes = _process_gradient_args(f, axes, coordinates, deltas)
    return tuple(
        first_derivative(f, axis=axis, **{kwarg: positions[ind]})
        for ind, axis in enumerate(axes)
    )


def laplacian(f, axes=None, coordinates=None, deltas=None):
    """Upstream `tools.laplacian`: the sum of the per-axis second derivatives."""
    f = mag(f)
    kwarg, positions, axes = _process_gradient_args(f, axes, coordinates, deltas)
    derivs = [
        second_derivative(f, axis=axis, **{kwarg: positions[ind]})
        for ind, axis in enumerate(axes)
    ]
    return sum(derivs)
