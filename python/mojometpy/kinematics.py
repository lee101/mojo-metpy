"""Port of `metpy/calc/kinematics.py` (MetPy 1.7.1), in upstream source order.

Signatures are upstream's.  `dx`/`dy` may be a scalar or a per-axis vector of
length `n-1`, which is what upstream's `first_derivative(delta=...)` accepts.
The map-factor correction is upstream's: both scales are differenced like any
other data field, and one of the two corrections is skipped unless BOTH scales
are given.
"""

from __future__ import annotations

import numpy as np

from ._ffi import axis_geometry, bcast, call, call2, call3, mag
from .basic import coriolis_parameter
from .tools import first_derivative, get_layer_heights



def _delta(value, shape, axis):
    """Broadcast a spacing onto the grid the way upstream's `first_derivative` does."""
    value = mag(value)
    if value.ndim == 0:
        return float(value)
    # upstream's `_broadcast_to_axis`: a 1-D spacing is moved onto the axis it
    # spaces, every other shape is already positioned
    if value.ndim == 1 and value.ndim < len(shape):
        new_shape = [1] * len(shape)
        new_shape[axis] = value.size
        value = value.reshape(new_shape)
    return np.ascontiguousarray(np.broadcast_to(value, tuple(shape)), dtype=np.float64)


def _axes(shape, x_dim, y_dim):
    nx, sx, ox = axis_geometry(shape, x_dim)
    ny, sy, oy = axis_geometry(shape, y_dim)
    # the three-point stencils read one point either side of every element, so
    # both axes need at least three points; upstream's `_process_deriv_args`
    # raises the same message before it ever slices
    if nx < 3 or ny < 3:
        raise ValueError("f must have at least 3 point along the desired axis.")
    return (nx, sx, ox), (ny, sy, oy)


def _vector_derivative(u, v, dx, dy, shape):
    u = np.ascontiguousarray(mag(u), dtype=np.float64)
    v = np.ascontiguousarray(mag(v), dtype=np.float64)
    (nx, sx, ox), (ny, sy, oy) = _axes(shape, -1, -2)
    ddx = _delta(dx, shape, len(shape) - 1)
    ddy = _delta(dy, shape, len(shape) - 2)
    dudx, dudy, dvdx, dvdy = (np.empty(shape, dtype=np.float64) for _ in range(4))
    if isinstance(ddx, float) and isinstance(ddy, float):
        # the scalar path keeps the kernels to a length-`n-1` spacing vector
        ddx = np.full(nx, ddx)
        ddy = np.full(ny, ddy)
    from . import _lib

    _lib.lib().mm_vector_derivative(
        u.ctypes.data, v.ctypes.data, ddx.ctypes.data, ddy.ctypes.data,
        nx, sx, ox, ny, sy, oy,
        dudx.ctypes.data, dudy.ctypes.data, dvdx.ctypes.data, dvdy.ctypes.data,
    )
    return dudx, dudy, dvdx, dvdy


def _derivatives(u, v, dx, dy, parallel_scale, meridional_scale, x_dim=-1, y_dim=-2):
    u, v = mag(u), mag(v)
    shape = np.broadcast_shapes(u.shape, v.shape)
    u = np.ascontiguousarray(np.broadcast_to(u, shape), dtype=np.float64)
    v = np.ascontiguousarray(np.broadcast_to(v, shape), dtype=np.float64)
    dudx, dudy, dvdx, dvdy = _vector_derivative(u, v, dx, dy, shape)
    return _map_correction(u, v, dudx, dudy, dvdx, dvdy, dx, dy, shape, x_dim, y_dim,
                           parallel_scale, meridional_scale)


def _map_correction(u, v, dudx, dudy, dvdx, dvdy, dx, dy, shape, x_dim, y_dim,
                    parallel_scale, meridional_scale):
    """Upstream's post-hoc map factor corrections, in its order."""
    if parallel_scale is None or meridional_scale is None:
        return dudx, dudy, dvdx, dvdy
    nx, _, _ = axis_geometry(shape, x_dim)
    ny, _, _ = axis_geometry(shape, y_dim)
    ps = np.ascontiguousarray(np.broadcast_to(mag(parallel_scale), shape), dtype=np.float64)
    ms = np.ascontiguousarray(np.broadcast_to(mag(meridional_scale), shape), dtype=np.float64)
    dpdy = first_derivative(ps, axis=y_dim, delta=dy)
    dmdx = first_derivative(ms, axis=x_dim, delta=dx)
    from ._ffi import size

    from . import _lib

    _lib.lib().mm_vector_derivative_map_correction(
        u.ctypes.data, v.ctypes.data, dpdy.ctypes.data, dmdx.ctypes.data,
        ps.ctypes.data, ms.ctypes.data, size(shape),
        dudx.ctypes.data, dudy.ctypes.data, dvdx.ctypes.data, dvdy.ctypes.data,
    )
    return dudx, dudy, dvdx, dvdy


def vector_derivative(
    u, v, *, dx=None, dy=None, x_dim=-1, y_dim=-2, parallel_scale=None,
    meridional_scale=None, return_only=None,
):
    """Upstream `tools.vector_derivative`."""
    if dx is None or dy is None:
        raise ValueError('Must specify either "dx" or "dy" for value positions.')
    dudx, dudy, dvdx, dvdy = _derivatives(u, v, dx, dy, parallel_scale, meridional_scale,
                                           x_dim, y_dim)
    derivatives = {"du/dx": dudx, "du/dy": dudy, "dv/dx": dvdx, "dv/dy": dvdy}
    if return_only is None:
        return (dudx, dudy, dvdx, dvdy)
    if isinstance(return_only, str):
        return derivatives[return_only]
    return tuple(derivatives[c] for c in return_only)


def geospatial_gradient(
    f, *, dx=None, dy=None, x_dim=-1, y_dim=-2, parallel_scale=None,
    meridional_scale=None, return_only=None,
):
    """Upstream `tools.geospatial_gradient`, map factor correction included."""
    f = mag(f)
    dfdx = first_derivative(f, axis=x_dim, delta=dx)
    dfdy = first_derivative(f, axis=y_dim, delta=dy)
    derivatives = {"df/dx": dfdx, "df/dy": dfdy}
    if parallel_scale is not None and meridional_scale is not None:
        derivatives["df/dx"] = dfdx * np.broadcast_to(mag(parallel_scale), dfdx.shape)
        derivatives["df/dy"] = dfdy * np.broadcast_to(mag(meridional_scale), dfdy.shape)
    if return_only is None:
        return (derivatives["df/dx"], derivatives["df/dy"])
    if isinstance(return_only, str):
        return derivatives[return_only]
    return tuple(derivatives[c] for c in return_only)


def vorticity(u, v, *, dx=None, dy=None, x_dim=-1, y_dim=-2, parallel_scale=None,
              meridional_scale=None):
    dudy, dvdx = vector_derivative(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
        return_only=("du/dy", "dv/dx"),
    )
    return call("mm_vorticity", dudy, dvdx, shape=dudy.shape)



def shear_vorticity(u, v, *, dx=None, dy=None, x_dim=-1, y_dim=-2,
                    parallel_scale=None, meridional_scale=None):
    dudx, dudy, dvdx, dvdy = vector_derivative(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
    )
    (a, b, c, e, f, g), shape = bcast(u, v, dudx, dudy, dvdx, dvdy)
    return call("mm_shear_vorticity", a, b, c, e, f, g, shape=shape)


def curvature_vorticity(u, v, *, dx=None, dy=None, x_dim=-1, y_dim=-2,
                        parallel_scale=None, meridional_scale=None):
    dudx, dudy, dvdx, dvdy = vector_derivative(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
    )
    (a, b, c, e, f, g), shape = bcast(u, v, dudx, dudy, dvdx, dvdy)
    return call("mm_curvature_vorticity", a, b, c, e, f, g, shape=shape)

def divergence(u, v, *, dx=None, dy=None, x_dim=-1, y_dim=-2, parallel_scale=None,
               meridional_scale=None):
    dudx, dvdy = vector_derivative(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
        return_only=("du/dx", "dv/dy"),
    )
    return call("mm_divergence", dudx, dvdy, shape=dudx.shape)


def shearing_deformation(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2, *,
                         parallel_scale=None, meridional_scale=None):
    dudy, dvdx = vector_derivative(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
        return_only=("du/dy", "dv/dx"),
    )
    return call("mm_shearing_deformation", dudy, dvdx, shape=dudy.shape)


def stretching_deformation(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2, *,
                           parallel_scale=None, meridional_scale=None):
    dudx, dvdy = vector_derivative(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
        return_only=("du/dx", "dv/dy"),
    )
    return call("mm_stretching_deformation", dudx, dvdy, shape=dudx.shape)


def total_deformation(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2, *,
                      parallel_scale=None, meridional_scale=None):
    dudx, dudy, dvdx, dvdy = vector_derivative(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
    )
    return call("mm_total_deformation", dudx, dudy, dvdx, dvdy, shape=dudx.shape)


def advection(scalar, u=None, v=None, w=None, *, dx=None, dy=None, dz=None, x_dim=-1,
              y_dim=-2, vertical_dim=-3, parallel_scale=None, meridional_scale=None):
    scalar = mag(scalar)
    # upstream builds the horizontal gradient from the components it was given,
    # in `u`, `v` order, and appends the vertical one only when `w` is present
    winds = {key: value for key, value in {"u": u, "v": v, "w": w}.items()
             if value is not None}
    horizontal = [value for key, value in {"u": "df/dx", "v": "df/dy"}.items()
                  if key in winds]
    gradient_vector = []
    if horizontal:
        gradient_vector = list(geospatial_gradient(
            scalar, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
            parallel_scale=parallel_scale, meridional_scale=meridional_scale,
            return_only=horizontal,
        ))
    if "w" in winds:
        gradient_vector.append(
            first_derivative(scalar, axis=vertical_dim, delta=dz)
        )
    arrays, shape = bcast(*winds.values())
    # the kernel always takes three wind components and three gradients, so the
    # absent ones are zero-filled and switched off with `use_w`
    zeros = np.zeros(shape, dtype=np.float64)
    u_buf, v_buf, w_buf = list(arrays) + [zeros] * (3 - len(arrays))
    dfdx, dfdy, dfdz = list(gradient_vector) + [zeros] * (3 - len(gradient_vector))
    return call(
        "mm_advection", dfdx, dfdy, dfdz, u_buf, v_buf, w_buf,
        1 if w is not None else 0, shape=shape,
    )

def frontogenesis(potential_temperature, u, v, dx=None, dy=None, x_dim=-1, y_dim=-2, *,
                  parallel_scale=None, meridional_scale=None):
    ddy_theta, ddx_theta = geospatial_gradient(
        potential_temperature, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
        return_only=("df/dy", "df/dx"),
    )
    shrd = shearing_deformation(
        u, v, dx, dy, x_dim=x_dim, y_dim=y_dim, parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
    )
    strd = stretching_deformation(
        u, v, dx, dy, x_dim=x_dim, y_dim=y_dim, parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
    )
    tdef = total_deformation(
        u, v, dx, dy, x_dim=x_dim, y_dim=y_dim, parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
    )
    div = divergence(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim, parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
    )
    return call(
        "mm_frontogenesis", ddx_theta, ddy_theta, shrd, strd, tdef, div, shape=div.shape
    )


def geostrophic_wind(height, dx=None, dy=None, latitude=None, x_dim=-1, y_dim=-2, *,
                     parallel_scale=None, meridional_scale=None):
    dhdx, dhdy = geospatial_gradient(
        height, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim, parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
    )
    (lat,), shape = bcast(latitude)
    return call2("mm_geostrophic_wind", dhdx, dhdy, lat, shape=shape)


def ageostrophic_wind(height, u, v, dx=None, dy=None, latitude=None, x_dim=-1, y_dim=-2,
                      *, parallel_scale=None, meridional_scale=None):
    u_g, v_g = geostrophic_wind(
        height, dx, dy, latitude, x_dim=x_dim, y_dim=y_dim, parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
    )
    (a, b), shape = bcast(u, u_g)
    (c, d), _ = bcast(v, v_g)
    return np.ascontiguousarray(a) - u_g, np.ascontiguousarray(c) - v_g


def montgomery_streamfunction(height, temperature):
    (z, t), shape = bcast(height, temperature)
    return call("mm_montgomery_streamfunction", z, t, shape=shape)


def absolute_vorticity(u, v, dx=None, dy=None, latitude=None, x_dim=-1, y_dim=-2, *,
                       parallel_scale=None, meridional_scale=None):
    f = coriolis_parameter(latitude)
    relative_vorticity = vorticity(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim, parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
    )
    return relative_vorticity + f


def storm_relative_helicity(height, u, v, depth, *, bottom=None, storm_u=None,
                            storm_v=None):
    """Upstream `kinematics.storm_relative_helicity`; heights in m, winds in m/s."""
    if bottom is None:
        bottom = 0.0
    if storm_u is None:
        storm_u = 0.0
    if storm_v is None:
        storm_v = 0.0
    _, u, v = get_layer_heights(mag(height), mag(depth), mag(u), mag(v),
                                with_agl=True, bottom=mag(bottom))
    return call3("mm_storm_relative_helicity", np.ascontiguousarray(u),
                 np.ascontiguousarray(v), float(mag(storm_u)), float(mag(storm_v)),
                 n=u.size)


def potential_vorticity_baroclinic(potential_temperature, pressure, u, v, dx=None,
                                   dy=None, latitude=None, x_dim=-1, y_dim=-2,
                                   vertical_dim=-3, *, parallel_scale=None,
                                   meridional_scale=None):
    potential_temperature = mag(potential_temperature)
    pressure = mag(pressure)
    if (potential_temperature.shape[vertical_dim] < 3
            or pressure.shape[vertical_dim] < 3
            or potential_temperature.shape[vertical_dim] != pressure.shape[vertical_dim]):
        raise ValueError("Length of potential temperature along the vertical axis "
                         f"{vertical_dim} must be at least 3.")
    avor = absolute_vorticity(u, v, dx, dy, latitude, x_dim=x_dim, y_dim=y_dim,
                              parallel_scale=parallel_scale,
                              meridional_scale=meridional_scale)
    dthetadp = first_derivative(potential_temperature, x=pressure, axis=vertical_dim)
    dthetady, dthetadx = geospatial_gradient(
        potential_temperature, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
    )
    dudp = first_derivative(mag(u), x=pressure, axis=vertical_dim)
    dvdp = first_derivative(mag(v), x=pressure, axis=vertical_dim)
    return call("mm_potential_vorticity_baroclinic", dudp, dthetady, dvdp, dthetadx,
                avor, dthetadp, shape=dthetadp.shape)


def potential_vorticity_barotropic(height, u, v, dx=None, dy=None, latitude=None,
                                    x_dim=-1, y_dim=-2, *, parallel_scale=None,
                                    meridional_scale=None):
    avor = absolute_vorticity(u, v, dx, dy, latitude, x_dim=x_dim, y_dim=y_dim,
                              parallel_scale=parallel_scale,
                              meridional_scale=meridional_scale)
    (a, b), shape = bcast(avor, height)
    return call("mm_potential_vorticity_barotropic", a, b, shape=shape)


def inertial_advective_wind(u, v, u_geostrophic, v_geostrophic, dx=None, dy=None,
                            latitude=None, x_dim=-1, y_dim=-2, *,
                            parallel_scale=None, meridional_scale=None):
    dugdx, dugdy = geospatial_gradient(
        u_geostrophic, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
    )


    dvgdx, dvgdy = geospatial_gradient(
        v_geostrophic, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
    )
    (uu, du, lat), shape = bcast(u, dugdx, latitude)
    (vv, dv, _), _ = bcast(v, dvgdx, latitude)
    return call2("mm_inertial_advective_wind", uu, vv, du, dugdy, dv, dvgdy, lat,
                 shape=shape)


def q_vector(u, v, temperature, pressure, dx=None, dy=None, static_stability=1,
             x_dim=-1, y_dim=-2, *, parallel_scale=None, meridional_scale=None):
    dudx, dudy, dvdx, dvdy = vector_derivative(
        u, v, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
    )
    dtempdx, dtempdy = geospatial_gradient(
        temperature, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
        parallel_scale=parallel_scale, meridional_scale=meridional_scale,
    )
    return call2("mm_q_vector", dudx, dudy, dvdx, dvdy, dtempdx, dtempdy, pressure,
                 float(mag(static_stability)), shape=dudx.shape)


def geospatial_laplacian(f, *, dx=None, dy=None, x_dim=-1, y_dim=-2,
                         parallel_scale=None, meridional_scale=None):
    grad_u, grad_y = geospatial_gradient(
        f, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim, parallel_scale=parallel_scale,
        meridional_scale=meridional_scale,
    )
    return divergence(grad_u, grad_y, dx=dx, dy=dy, x_dim=x_dim, y_dim=y_dim,
                      parallel_scale=parallel_scale,
                      meridional_scale=meridional_scale)
