"""Port of `metpy/calc/kinematics.py` (MetPy 1.7.1), in upstream source order.

`vector_derivative` keeps upstream's component-by-component dispatch, with
`dx`/`dy` reduced to the `first_derivative` delta of the corresponding axis.
Upstream's map-factor correction (`parallel_scale` / `meridional_scale`)
is ported: the two scale fields are differenced like any other data field and
the four Cartesian derivatives are corrected in place afterwards.
"""

from max.algorithm import parallelize
from std.math import atan2, cos, sin, sqrt
from mojometpy.constants import Rd, g, omega

from mojometpy.simd import F, PAR_MIN, PAR_WORKERS, W
from mojometpy.thermo import dry_static_energy
from mojometpy.tools import first_derivative
from mojometpy.types import Ptr


def vector_derivative(
    u: Ptr,
    v: Ptr,
    dx: Ptr,
    dy: Ptr,
    nx: Int,
    x_stride: Int,
    x_outer: Int,
    ny: Int,
    y_stride: Int,
    y_outer: Int,
    out_dudx: Ptr,
    out_dudy: Ptr,
    out_dvdx: Ptr,
    out_dvdy: Ptr,
):
    first_derivative(u, dx, nx, x_stride, x_outer, out_dudx)
    first_derivative(u, dy, ny, y_stride, y_outer, out_dudy)
    first_derivative(v, dx, nx, x_stride, x_outer, out_dvdx)
    first_derivative(v, dy, ny, y_stride, y_outer, out_dvdy)

# Upstream applies the map factor corrections to the Cartesian derivatives it
# has already computed, in this order, so the four outputs are corrected in
# place here rather than folded into the stencils.  Both scales are fields, not
# scalars: they are multiplied element for element exactly as upstream does.
def vector_derivative_map_correction(
    u: Ptr,
    v: Ptr,
    dpdy: Ptr,
    dmdx: Ptr,
    parallel_scale: Ptr,
    meridional_scale: Ptr,
    n: Int,
    out_dudx: Ptr,
    out_dudy: Ptr,
    out_dvdx: Ptr,
    out_dvdy: Ptr,
):
    var i = 0
    while i + W <= n:
        var uv = u.unsafe_load[width=W](i)
        var vv = v.unsafe_load[width=W](i)
        var ps = parallel_scale.unsafe_load[width=W](i)
        var ms = meridional_scale.unsafe_load[width=W](i)
        var dxc = ms / ps * dpdy.unsafe_load[width=W](i)
        var dyc = ps / ms * dmdx.unsafe_load[width=W](i)
        out_dudx.unsafe_store(i, ps * out_dudx.unsafe_load[width=W](i) - vv * dxc)
        out_dudy.unsafe_store(i, ms * out_dudy.unsafe_load[width=W](i) + vv * dyc)
        out_dvdx.unsafe_store(i, ps * out_dvdx.unsafe_load[width=W](i) + uv * dxc)
        out_dvdy.unsafe_store(i, ms * out_dvdy.unsafe_load[width=W](i) - uv * dyc)
        i += W
    while i < n:
        var uv = u[unsafe_offset=i]
        var vv = v[unsafe_offset=i]
        var ps = parallel_scale[unsafe_offset=i]
        var ms = meridional_scale[unsafe_offset=i]
        var dxc = ms / ps * dpdy[unsafe_offset=i]
        var dyc = ps / ms * dmdx[unsafe_offset=i]
        out_dudx[unsafe_offset=i] = ps * out_dudx[unsafe_offset=i] - vv * dxc
        out_dudy[unsafe_offset=i] = ms * out_dudy[unsafe_offset=i] + vv * dyc
        out_dvdx[unsafe_offset=i] = ps * out_dvdx[unsafe_offset=i] + uv * dxc
        out_dvdy[unsafe_offset=i] = ms * out_dvdy[unsafe_offset=i] - uv * dyc
        i += 1


def vorticity(
    dudy: Ptr,
    dvdx: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i, dvdx.unsafe_load[width=W](i) - dudy.unsafe_load[width=W](i)
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = dvdx[unsafe_offset=i] - dudy[unsafe_offset=i]
        i += 1


# ---- kinematics.py:96 ----


def shear_vorticity(
    u: Ptr,
    v: Ptr,
    dudx: Ptr,
    dudy: Ptr,
    dvdx: Ptr,
    dvdy: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        var uv = u.unsafe_load[width=W](i)
        var vv = v.unsafe_load[width=W](i)
        dst.unsafe_store(
            i,
            (vv * uv * dudx.unsafe_load[width=W](i)
             + vv * vv * dvdx.unsafe_load[width=W](i)
             - uv * uv * dudy.unsafe_load[width=W](i)
             - uv * vv * dvdy.unsafe_load[width=W](i))
            / (uv * uv + vv * vv),
        )
        i += W
    while i < n:
        var uv = u[unsafe_offset=i]
        var vv = v[unsafe_offset=i]
        dst[unsafe_offset=i] = (
            vv * uv * dudx[unsafe_offset=i]
            + vv * vv * dvdx[unsafe_offset=i]
            - uv * uv * dudy[unsafe_offset=i]
            - uv * vv * dvdy[unsafe_offset=i]
        ) / (uv * uv + vv * vv)
        i += 1


def divergence(
    dudx: Ptr,
    dvdy: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i, dudx.unsafe_load[width=W](i) + dvdy.unsafe_load[width=W](i)
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = dudx[unsafe_offset=i] + dvdy[unsafe_offset=i]
        i += 1


# ---- kinematics.py:174 ----


def curvature_vorticity(
    u: Ptr,
    v: Ptr,
    dudx: Ptr,
    dudy: Ptr,
    dvdx: Ptr,
    dvdy: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        var uv = u.unsafe_load[width=W](i)
        var vv = v.unsafe_load[width=W](i)
        dst.unsafe_store(
            i,
            (uv * uv * dvdx.unsafe_load[width=W](i)
             - vv * vv * dudy.unsafe_load[width=W](i)
             - vv * uv * dudx.unsafe_load[width=W](i)
             + uv * vv * dvdy.unsafe_load[width=W](i))
            / (uv * uv + vv * vv),
        )
        i += W
    while i < n:
        var uv = u[unsafe_offset=i]
        var vv = v[unsafe_offset=i]
        dst[unsafe_offset=i] = (
            uv * uv * dvdx[unsafe_offset=i]
            - vv * vv * dudy[unsafe_offset=i]
            - vv * uv * dudx[unsafe_offset=i]
            + uv * vv * dvdy[unsafe_offset=i]
        ) / (uv * uv + vv * vv)
        i += 1


def shearing_deformation(
    dudy: Ptr,
    dvdx: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i, dvdx.unsafe_load[width=W](i) + dudy.unsafe_load[width=W](i)
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = dvdx[unsafe_offset=i] + dudy[unsafe_offset=i]
        i += 1


def stretching_deformation(
    dudx: Ptr,
    dvdy: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i, dudx.unsafe_load[width=W](i) - dvdy.unsafe_load[width=W](i)
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = dudx[unsafe_offset=i] - dvdy[unsafe_offset=i]
        i += 1


def total_deformation(
    dudx: Ptr,
    dudy: Ptr,
    dvdx: Ptr,
    dvdy: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        var a = dvdx.unsafe_load[width=W](i) + dudy.unsafe_load[width=W](i)
        var b = dudx.unsafe_load[width=W](i) - dvdy.unsafe_load[width=W](i)
        dst.unsafe_store(i, sqrt(a * a + b * b))
        i += W
    while i < n:
        var a = dvdx[unsafe_offset=i] + dudy[unsafe_offset=i]
        var b = dudx[unsafe_offset=i] - dvdy[unsafe_offset=i]
        dst[unsafe_offset=i] = sqrt(a * a + b * b)
        i += 1


def advection(
    dfdx: Ptr,
    dfdy: Ptr,
    dfdz: Ptr,
    u: Ptr,
    v: Ptr,
    w: Ptr,
    use_w: Int,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        var total = u.unsafe_load[width=W](i) * dfdx.unsafe_load[width=W](i) + v.unsafe_load[
            width=W
        ](i) * dfdy.unsafe_load[width=W](i)
        if use_w == 1:
            total += w.unsafe_load[width=W](i) * dfdz.unsafe_load[width=W](i)
        dst.unsafe_store(i, -total)
        i += W
    while i < n:
        var total = u[unsafe_offset=i] * dfdx[unsafe_offset=i] + v[unsafe_offset=i] * dfdy[
            unsafe_offset=i
        ]
        if use_w == 1:
            total += w[unsafe_offset=i] * dfdz[unsafe_offset=i]
        dst[unsafe_offset=i] = -total
        i += 1


def _frontogenesis_chunk(
    ddx_theta: Ptr,
    ddy_theta: Ptr,
    shrd: Ptr,
    strd: Ptr,
    tdef: Ptr,
    div: Ptr,
    lo: Int,
    hi: Int,
    dst: Ptr,
):
    var n = hi
    for i in range(lo, hi):
        var mag_theta = sqrt(
            ddx_theta[unsafe_offset=i] ** 2 + ddy_theta[unsafe_offset=i] ** 2
        )
        var psi = 0.5 * atan2(shrd[unsafe_offset=i], strd[unsafe_offset=i])
        var sin_beta = 0.0
        if mag_theta != 0.0:
            sin_beta = (
                ddx_theta[unsafe_offset=i] * cos(psi) + ddy_theta[unsafe_offset=i] * sin(psi)
            ) / mag_theta
        dst[unsafe_offset=i] = 0.5 * mag_theta * (
            tdef[unsafe_offset=i] * (1 - 2 * sin_beta**2) - div[unsafe_offset=i]
        )


def frontogenesis(
    ddx_theta: Ptr,
    ddy_theta: Ptr,
    shrd: Ptr,
    strd: Ptr,
    tdef: Ptr,
    div: Ptr,
    n: Int,
    dst: Ptr,
):
    if n < PAR_MIN:
        _frontogenesis_chunk(ddx_theta, ddy_theta, shrd, strd, tdef, div, 0, n, dst)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _frontogenesis_chunk(ddx_theta, ddy_theta, shrd, strd, tdef, div, k * span, min((k + 1) * span, n), dst)
        parallelize(work, nw, nw)


def _geostrophic_wind_chunk(
    dhdx: Ptr,
    dhdy: Ptr,
    latitude: Ptr,
    lo: Int,
    hi: Int,
    out_u: Ptr,
    out_v: Ptr,
):
    var n = hi
    for i in range(lo, hi):
        var f = 2.0 * omega * sin(latitude[unsafe_offset=i])
        var norm_factor = g / f
        out_u[unsafe_offset=i] = -norm_factor * dhdy[unsafe_offset=i]
        out_v[unsafe_offset=i] = norm_factor * dhdx[unsafe_offset=i]


def geostrophic_wind(
    dhdx: Ptr,
    dhdy: Ptr,
    latitude: Ptr,
    n: Int,
    out_u: Ptr,
    out_v: Ptr,
):
    if n < PAR_MIN:
        _geostrophic_wind_chunk(dhdx, dhdy, latitude, 0, n, out_u, out_v)
    else:
        var nw = PAR_WORKERS
        var span = (n + nw - 1) // nw
        def work(k: Int) {imm}:
            _geostrophic_wind_chunk(dhdx, dhdy, latitude, k * span, min((k + 1) * span, n), out_u, out_v)
        parallelize(work, nw, nw)


def montgomery_streamfunction(height: Ptr, temperature: Ptr, n: Int, dst: Ptr):
    dry_static_energy(height, temperature, n, dst)


# ---- kinematics.py:927 ----


def storm_relative_helicity(
    u: Ptr, v: Ptr, storm_u: Float64, storm_v: Float64, n: Int, dst: Ptr
):
    # `int_layers[i] = u[i+1] * v[i] - u[i] * v[i+1]` on the storm-relative
    # wind; upstream splits the sum by sign and returns both halves and their
    # sum, and an empty half is zero, which an untouched accumulator gives.
    var positive = 0.0
    var negative = 0.0
    for i in range(1, n):
        var u_now = u[unsafe_offset=i] - storm_u
        var v_now = v[unsafe_offset=i] - storm_v
        var u_prev = u[unsafe_offset=i - 1] - storm_u
        var v_prev = v[unsafe_offset=i - 1] - storm_v
        var layer = u_now * v_prev - u_prev * v_now
        if layer > 0.0:
            positive += layer
        else:
            negative += layer
    dst[unsafe_offset=0] = positive
    dst[unsafe_offset=1] = negative
    dst[unsafe_offset=2] = positive + negative


def potential_vorticity_baroclinic(
    dudp: Ptr,
    dthetady: Ptr,
    dvdp: Ptr,
    dthetadx: Ptr,
    avor: Ptr,
    dthetadp: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(
            i,
            -F(g)
            * (
                dudp.unsafe_load[width=W](i) * dthetady.unsafe_load[width=W](i)
                - dvdp.unsafe_load[width=W](i) * dthetadx.unsafe_load[width=W](i)
                + avor.unsafe_load[width=W](i) * dthetadp.unsafe_load[width=W](i)
            ),
        )
        i += W
    while i < n:
        dst[unsafe_offset=i] = -g * (
            dudp[unsafe_offset=i] * dthetady[unsafe_offset=i]
            - dvdp[unsafe_offset=i] * dthetadx[unsafe_offset=i]
            + avor[unsafe_offset=i] * dthetadp[unsafe_offset=i]
        )
        i += 1


def potential_vorticity_barotropic(avor: Ptr, height: Ptr, n: Int, dst: Ptr):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(i, avor.unsafe_load[width=W](i) / height.unsafe_load[width=W](i))
        i += W
    while i < n:
        dst[unsafe_offset=i] = avor[unsafe_offset=i] / height[unsafe_offset=i]
        i += 1


def inertial_advective_wind(
    u: Ptr,
    v: Ptr,
    dugdx: Ptr,
    dugdy: Ptr,
    dvgdx: Ptr,
    dvgdy: Ptr,
    latitude: Ptr,
    n: Int,
    out_u: Ptr,
    out_v: Ptr,
):
    var i = 0
    while i + W <= n:
        var uv = u.unsafe_load[width=W](i)
        var vv = v.unsafe_load[width=W](i)
        var f = F(2.0 * omega) * sin(latitude.unsafe_load[width=W](i))
        out_u.unsafe_store(
            i,
            -(uv * dvgdx.unsafe_load[width=W](i) + vv * dvgdy.unsafe_load[width=W](i)) / f,
        )
        out_v.unsafe_store(
            i,
            (uv * dugdx.unsafe_load[width=W](i) + vv * dugdy.unsafe_load[width=W](i)) / f,
        )
        i += W
    while i < n:
        var uv = u[unsafe_offset=i]
        var vv = v[unsafe_offset=i]
        var f = 2.0 * omega * sin(latitude[unsafe_offset=i])
        out_u[unsafe_offset=i] = -(uv * dvgdx[unsafe_offset=i] + vv * dvgdy[unsafe_offset=i]) / f
        out_v[unsafe_offset=i] = (uv * dugdx[unsafe_offset=i] + vv * dugdy[unsafe_offset=i]) / f
        i += 1


def q_vector(
    dudx: Ptr,
    dudy: Ptr,
    dvdx: Ptr,
    dvdy: Ptr,
    dtempdx: Ptr,
    dtempdy: Ptr,
    pressure: Ptr,
    static_stability: Float64,
    n: Int,
    out_q1: Ptr,
    out_q2: Ptr,
):
    var i = 0
    while i + W <= n:
        var scale = F(-Rd / (static_stability))
        var dtempdxv = dtempdx.unsafe_load[width=W](i)
        var dtempdyv = dtempdy.unsafe_load[width=W](i)
        out_q1.unsafe_store(
            i,
            scale
            / pressure.unsafe_load[width=W](i)
            * (
                dudx.unsafe_load[width=W](i) * dtempdxv
                + dvdx.unsafe_load[width=W](i) * dtempdyv
            ),
        )
        out_q2.unsafe_store(
            i,
            scale
            / pressure.unsafe_load[width=W](i)
            * (
                dudy.unsafe_load[width=W](i) * dtempdxv
                + dvdy.unsafe_load[width=W](i) * dtempdyv
            ),
        )
        i += W
    while i < n:
        var scale = -Rd / (pressure[unsafe_offset=i] * static_stability)
        var dtempdxv = dtempdx[unsafe_offset=i]
        var dtempdyv = dtempdy[unsafe_offset=i]
        out_q1[unsafe_offset=i] = scale * (
            dudx[unsafe_offset=i] * dtempdxv + dvdx[unsafe_offset=i] * dtempdyv
        )
        out_q2[unsafe_offset=i] = scale * (
            dudy[unsafe_offset=i] * dtempdxv + dvdy[unsafe_offset=i] * dtempdyv
        )
        i += 1
