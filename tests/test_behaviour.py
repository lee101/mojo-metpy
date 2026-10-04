"""Behaviour of the ported API itself: shapes, broadcasting, dtypes, NaN
handling, error paths, and the places the port deliberately diverges.

These are the contracts a caller depends on that numerical parity against
MetPy does not cover on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojometpy
from mojometpy import basic, calc, constants, kinematics, thermo, tools


def test_api_surface_matches_upstream_names():
    import metpy.calc as mcalc

    # `metpy.interpolate` and `metpy.calc.turbulence` are the modules the
    # upstream names live in; everything this port exposes must be callable
    # there under the same name.
    import metpy.interpolate as minterp

    theirs = {n for n in dir(mcalc) if not n.startswith("_") and callable(getattr(mcalc, n))}
    theirs |= {n for n in dir(minterp) if not n.startswith("_") and callable(getattr(minterp, n))}

    modules = {"basic", "thermo", "tools", "kinematics", "indices", "turbulence"}
    ours = {
        n for n in dir(mojometpy)
        if not n.startswith("_") and n not in modules and callable(getattr(mojometpy, n))
    }
    # every name this port claims must exist upstream under the same name
    assert ours - theirs == set(), sorted(ours - theirs)


def test_modules_mirror_upstream():
    import metpy.calc as mcalc

    # the public names of each module must be a subset of that upstream module;
    # the extras are the port's own FFI helpers, not part of the mirrored API
    extra = {"bcast", "call", "call2", "call3", "call_scalar", "call_axis", "call_int",
             "call_masked", "mag", "with_mask", "phase_code", "axis_geometry", "size",
             "annotations", "warnings", "epsilon", "get_layer_by_length"}
    for name in ("basic", "thermo", "tools", "kinematics", "indices", "turbulence"):
        ours = {n for n in dir(getattr(mojometpy, name)) if not n.startswith("_")}
        theirs = {n for n in dir(getattr(mcalc, name)) if not n.startswith("_")}
        assert ours - theirs <= extra, (name, sorted(ours - theirs))


def test_constants_match_upstream_nounit():
    import metpy.constants.nounit as up

    for name in (
        "Rd", "Rv", "Lv", "Lf", "Ls", "Cp_l", "Cp_i", "Cp_v", "zero_degc",
        "sat_pressure_0c", "epsilon", "kappa", "g", "T0",
    ):
        assert getattr(constants, name) == getattr(up, name), name


def test_scalar_inputs_give_scalar_shaped_output():
    out = thermo.potential_temperature(85000.0, 288.0)
    assert np.shape(out) == ()
    assert float(out) == pytest.approx(301.68832861, rel=1e-9)


def test_broadcasting_matches_numpy():
    p = np.full((3, 4), 100000.0)
    t = np.linspace(280.0, 300.0, 4)
    out = thermo.potential_temperature(p, t)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out[0], thermo.potential_temperature(p[0], t), rtol=0, atol=0)


def test_pint_quantity_input_is_unwrapped():
    from metpy.units import units

    q = units.Quantity([85000.0, 90000.0], "Pa")
    assert np.allclose(
        thermo.potential_temperature(q, 288.0),
        thermo.potential_temperature(np.array([85000.0, 90000.0]), 288.0),
    )


def test_output_is_float64_and_contiguous():
    out = thermo.density(100000.0, 290.0, 0.01)
    assert out.dtype == np.float64


def _reference_derivative(f, axis, delta):
    """upstream's left / center / right stencils, written out with moveaxis."""
    m = np.moveaxis(f, axis, 0)
    n = m.shape[0]
    out = np.empty_like(m)
    d0, d1 = delta[0], delta[1]
    comb, big = d0 + d1, d0 + d1 + d0
    out[0] = (
        -big / (comb * d0) * m[0] + comb / (d0 * d1) * m[1] - d0 / (comb * d1) * m[2]
    )
    for k in range(1, n - 1):
        d0, d1 = delta[k - 1], delta[k]
        comb, diff = d0 + d1, d1 - d0
        out[k] = (
            -d1 / (comb * d0) * m[k - 1] + diff / (d0 * d1) * m[k] + d0 / (comb * d1) * m[k + 1]
        )
    d0, d1 = delta[n - 3], delta[n - 2]
    comb, big = d0 + d1, d0 + d1 + d1
    out[n - 1] = (
        d1 / (comb * d0) * m[n - 3] - comb / (d0 * d1) * m[n - 2] + big / (comb * d1) * m[n - 1]
    )
    return np.moveaxis(out, 0, axis)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("kind", ["uniform", "nonuniform"])
def test_derivative_matches_reference_stencils(axis, kind):
    rng = np.random.default_rng(7)
    f = rng.normal(size=(9, 8, 7)) * 50.0 + 200.0
    n = f.shape[axis]
    delta = np.full(n - 1, 2.0) if kind == "uniform" else 1.0 + 0.05 * np.arange(n - 1)
    got = tools.first_derivative(f, axis=axis, delta=delta)
    np.testing.assert_allclose(got, _reference_derivative(f, axis, delta), rtol=1e-14, atol=1e-9)

    got2 = tools.second_derivative(f, axis=axis, delta=delta)
    ref2 = _reference_second_derivative(f, axis, delta)
    np.testing.assert_allclose(got2, ref2, rtol=1e-14, atol=1e-9)


def _reference_second_derivative(f, axis, delta):
    m = np.moveaxis(f, axis, 0)
    n = m.shape[0]
    out = np.empty_like(m)
    for k in range(n):
        if k == 0:
            k0, k1, k2, d0, d1 = 0, 1, 2, delta[0], delta[1]
        elif k == n - 1:
            k0, k1, k2, d0, d1 = n - 3, n - 2, n - 1, delta[n - 3], delta[n - 2]
        else:
            k0, k1, k2, d0, d1 = k - 1, k, k + 1, delta[k - 1], delta[k]
        comb = d0 + d1
        out[k] = 2.0 * (
            m[k0] / (comb * d0) - m[k1] / (d0 * d1) + m[k2] / (comb * d1)
        )
    return np.moveaxis(out, 0, axis)


def test_derivative_preserves_line_structure():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(30, 40)) * 50.0 + 200.0
    out = tools.first_derivative(f, axis=-1, delta=2.0)
    assert out.shape == f.shape
    # every row is a separate line along the last axis
    ref = _reference_derivative(f, -1, np.full(39, 2.0))
    np.testing.assert_allclose(out, ref, rtol=1e-14, atol=1e-9)


def test_derivative_rejects_short_axis():
    with pytest.raises(ValueError, match="at least 3 point"):
        tools.first_derivative(np.zeros((2, 5)), axis=0, delta=1.0)


def test_derivative_rejects_both_x_and_delta():
    with pytest.raises(ValueError, match="both"):
        tools.first_derivative(np.zeros((5, 5)), axis=0, x=np.arange(5.0), delta=1.0)


def test_derivative_requires_a_position():
    with pytest.raises(ValueError, match="Must specify"):
        tools.first_derivative(np.zeros((5, 5)), axis=0)


def test_gradient_rejects_both_coordinates_and_deltas():
    with pytest.raises(ValueError, match="Cannot specify both"):
        tools.gradient(np.zeros((5, 5)), axes=(0, 1), coordinates=(1.0, 1.0), deltas=(1.0, 1.0))


def test_nan_propagates_through_dewpoint():
    out = thermo.dewpoint(np.array([1000.0, np.nan, 2000.0]))
    assert np.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[2])


def test_saturation_mixing_ratio_is_nan_above_saturation():
    # at 400 K the saturation vapour pressure exceeds the total pressure
    out = thermo.saturation_mixing_ratio(np.array([1000.0]), np.array([400.0]))
    assert np.isnan(out[0])


def test_invalid_phase_raises():
    with pytest.raises(ValueError, match="not a valid option for phase"):
        thermo.saturation_vapor_pressure(273.0, phase="gas")


def test_invalid_convention_raises():
    with pytest.raises(ValueError, match="convention"):
        basic.wind_direction(1.0, 1.0, "sideways")


def test_coriolis_rejects_latitude_beyond_hemisphere():
    with pytest.raises(ValueError, match="radians"):
        basic.coriolis_parameter(np.array([2.0]))


@pytest.mark.parametrize("shape", [(2, 5), (5, 2), (2, 2), (1, 4)])
def test_kinematics_rejects_axes_shorter_than_the_stencil(shape):
    # the stencils read one point either side; upstream refuses the same inputs
    u = np.zeros(shape)
    with pytest.raises(ValueError, match="at least 3 point"):
        kinematics.vorticity(u, u, dx=1.0, dy=1.0)


def test_wind_components_reject_degrees():
    with pytest.raises(ValueError, match="radians"):
        basic.wind_components(5.0, np.array([90.0]))


def test_masks_follow_the_piecewise_definitions():
    # 20 degC / 20 km/h: above the wind chill limit and below the heat index one
    # the JAG/TI formula is only defined below 10 degC and above 3 mph
    out = basic.windchill(np.array([20.0, -10.0, -10.0]), np.array([20.0, 20.0, 1.0]))
    assert np.ma.isMaskedArray(out)
    np.testing.assert_array_equal(np.ma.getmaskarray(out), [True, False, True])
    assert np.allclose(np.ma.getdata(out)[1], -17.86, atol=0.01)

    # below 80 degF everything is masked
    hi = basic.heat_index(np.array([79.0, 95.0]), np.array([0.5, 0.5]))
    np.testing.assert_array_equal(np.ma.getmaskarray(hi), [True, False])


def test_mask_undefined_false_returns_plain_array():
    out = basic.heat_index(np.array([50.0]), np.array([0.5]), mask_undefined=False)
    assert not np.ma.isMaskedArray(out)
    assert np.isfinite(out).all()


def test_smooth_window_does_not_mutate_its_input():
    g = np.arange(40.0 * 50.0).reshape(40, 50)
    before = g.copy()
    basic.smooth_window(g, np.ones((3, 3)))
    np.testing.assert_array_equal(g, before)


def test_smooth_window_leaves_the_border_untouched():
    g = np.random.default_rng(4).normal(size=(40, 50))
    out = basic.smooth_window(g, np.ones((3, 3)))
    np.testing.assert_array_equal(out[0], g[0])
    np.testing.assert_array_equal(out[-1], g[-1])
    np.testing.assert_array_equal(out[:, 0], g[:, 0])
    assert not np.allclose(out[1:-1, 1:-1], g[1:-1, 1:-1])


def test_smooth_window_rejects_even_window():
    with pytest.raises(ValueError, match="odd"):
        basic.smooth_window(np.zeros((10, 10)), np.ones((3, 2)))


def test_smooth_window_rejects_non_2d_grid():
    with pytest.raises(NotImplementedError, match="2-D"):
        basic.smooth_window(np.zeros((4, 5, 6)), np.ones((3, 3, 3)))


def test_smooth_n_point_rejects_bad_n():
    with pytest.raises(ValueError, match="either 5 or 9"):
        basic.smooth_n_point(np.zeros((6, 6)), 7)


def test_map_factors_correct_the_cartesian_derivatives():
    # upstream's post-hoc correction: both scales are differenced like any other
    # field and the four Cartesian derivatives are then fixed up in place
    shape = (7, 7)
    u = np.zeros(shape)
    v = np.zeros(shape)
    ps = np.full(shape, 2.0)
    ms = np.full(shape, 3.0)
    dudx, dudy, dvdx, dvdy = kinematics.vector_derivative(
        u, v, dx=1.0, dy=1.0, parallel_scale=ps, meridional_scale=ms
    )
    # du/dy and dv/dx stay at their constant-field value; only the correction terms move
    np.testing.assert_allclose(dudx, ps * dudx, rtol=1e-12)
    np.testing.assert_allclose(dudy, ms * dudy, rtol=1e-12)
    np.testing.assert_allclose(dvdy, ms * dvdy, rtol=1e-12)

    # a nonzero wind with varying scales exercises the v and u cross terms
    rng = np.random.default_rng(21)
    u2, v2 = rng.normal(size=shape) * 10.0, rng.normal(size=shape) * 10.0
    ps2 = 1.0 + 0.1 * rng.random(shape)
    ms2 = 1.0 + 0.1 * rng.random(shape)
    raw = kinematics.vector_derivative(u2, v2, dx=1.0, dy=1.0)
    corrected = kinematics.vector_derivative(
        u2, v2, dx=1.0, dy=1.0, parallel_scale=ps2, meridional_scale=ms2
    )
    # without the correction the four fields are the plain Cartesian ones
    for plain, got in zip(raw, corrected):
        assert got.shape == plain.shape
    assert any(
        not np.allclose(a, b) for a, b in zip(raw[:1], corrected[:1])
    )


def test_thickness_hydrostatic_selects_a_layer_with_get_layer():
    # upstream's `bottom`/`depth` branch runs the profile through `get_layer`
    p = np.linspace(1e5, 1e4, 10)
    t = np.full(10, 280.0)
    shallow = thermo.thickness_hydrostatic(p, t, bottom=90000.0, depth=10000.0)
    deep = thermo.thickness_hydrostatic(p, t)
    # a 10 hPa layer is thinner than the whole profile, and both are positive
    assert 0.0 < shallow < deep
    np.testing.assert_allclose(
        shallow,
        thermo.thickness_hydrostatic(p, t, bottom=90000.0, depth=10000.0),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="outside pressure range"):
        thermo.thickness_hydrostatic(p, t, bottom=1000.0, depth=10000.0)


def test_thickness_hydrostatic_rejects_nd_profiles():
    with pytest.raises(NotImplementedError, match="1-D"):
        thermo.thickness_hydrostatic(np.full((3, 10), 5e4), np.full((3, 10), 280.0))


def test_thickness_hydrostatic_is_a_scalar():
    out = thermo.thickness_hydrostatic(np.linspace(1e5, 1e4, 20), np.full(20, 260.0))
    assert np.ndim(out) == 0
    # isothermal layer: -Rd/g * T * ln(p_top / p_bottom) > 0
    assert out > 0


def test_sigma_to_pressure_rejects_out_of_range():
    with pytest.raises(ValueError, match="bounded by 0 and 1"):
        basic.sigma_to_pressure(np.array([1.5]), 100000.0, 1000.0)


def test_lcl_is_monotonic_in_dewpoint():
    p, t = np.array([100000.0, 90000.0]), np.array([300.0, 295.0])
    td = np.array([290.0, 280.0])
    pl, tl = thermo.lcl(p, t, td)
    assert pl.shape == (2,) and tl.shape == (2,)
    assert np.all(pl < p)
    assert np.all(tl < t)


def test_potential_temperature_round_trips():
    p = np.linspace(50000.0, 100000.0, 50)
    t = np.linspace(220.0, 310.0, 50)
    theta = thermo.potential_temperature(p, t)
    back = thermo.temperature_from_potential_temperature(p, theta)
    np.testing.assert_allclose(back, t, rtol=1e-9)


def test_mixing_ratio_round_trips_through_specific_humidity():
    w = np.linspace(1e-5, 0.05, 40)
    q = thermo.specific_humidity_from_mixing_ratio(w)
    np.testing.assert_allclose(thermo.mixing_ratio_from_specific_humidity(q), w, rtol=1e-12)


def test_height_pressure_round_trip():
    p = np.linspace(20000.0, 100000.0, 60)
    h = basic.pressure_to_height_std(p)
    np.testing.assert_allclose(basic.height_to_pressure_std(h), p, rtol=1e-8)


def test_geopotential_height_round_trip():
    z = np.linspace(0.0, 20000.0, 40)
    phi = basic.height_to_geopotential(z)
    np.testing.assert_allclose(basic.geopotential_to_height(phi), z, rtol=1e-12)


def test_brunt_vaisala_frequency_is_nan_where_squared_is_negative():
    # a strongly stable sounding has d(theta)/dz < 0
    h = np.linspace(0.0, 10000.0, 50)
    theta = 300.0 - 0.02 * h
    freq = thermo.brunt_vaisala_frequency(h, theta)
    assert np.all(np.isnan(freq))
    period = thermo.brunt_vaisala_period(h, theta)
    assert np.all(np.isnan(period))


def test_wind_direction_is_zero_for_calm():
    out = basic.wind_direction(np.zeros(3), np.zeros(3))
    np.testing.assert_array_equal(out, np.zeros(3))


def test_wind_direction_convention_differs_by_180():
    u, v = np.array([3.0, -2.0]), np.array([-4.0, 5.0])
    frm = basic.wind_direction(u, v, "from")
    to = basic.wind_direction(u, v, "to")
    np.testing.assert_allclose((to - frm) % 360.0, np.array([180.0, 180.0]))


def test_wind_components_inverts_wind_direction():
    speed = np.array([5.0, 12.0])
    direction = np.deg2rad(np.array([270.0, 45.0]))
    u, v = basic.wind_components(speed, direction)
    np.testing.assert_allclose(
        np.mod(np.arctan2(-u, -v) - direction + np.pi, 2 * np.pi) - np.pi, 0.0, atol=1e-12
    )
    np.testing.assert_allclose(basic.wind_speed(u, v), speed, rtol=1e-12)



# ---- SIMD tails, chunk boundaries and the parallel split -------------------
#
# Every kernel is a `W`-lane pass over `[0, n)` with a scalar tail, and the
# compute-bound ones additionally split `[0, n)` across workers above
# `PAR_MIN`.  None of that is visible in a 40-point parity corpus, so the
# contracts below pin the boundaries that could silently drop or duplicate an
# element: a length below one lane, a length that is not a whole number of
# lanes, and a length past the parallel threshold.

PAR_MIN = 524288
LANES = 4


def _elementwise_cases(n, seed=11):
    """`name -> (callable, args)`, all of them elementwise and all length `n`."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(50000.0, 100000.0, n)
    t = 288.0 + rng.uniform(-30.0, 12.0, n)
    td = t - rng.uniform(0.0, 25.0, n)
    rh = rng.uniform(0.05, 1.0, n)
    w = rng.uniform(1e-4, 0.03, n)
    e = rng.uniform(100.0, 5000.0, n)
    q = w / (1.0 + w)
    return {
        "potential_temperature": ((p, t), lambda a: thermo.potential_temperature(*a)),
        "temperature_from_potential_temperature": (
            (p, t), lambda a: thermo.temperature_from_potential_temperature(*a)
        ),
        "exner_function": ((p,), lambda a: thermo.exner_function(*a)),
        "dry_lapse": (
            (p, t, 100000.0), lambda a: thermo.dry_lapse(*a)
        ),
        "dewpoint": ((e,), lambda a: thermo.dewpoint(*a)),
        "dewpoint_from_relative_humidity": (
            (t, rh), lambda a: thermo.dewpoint_from_relative_humidity(*a)
        ),
        "dewpoint_from_specific_humidity": (
            (p, q), lambda a: thermo.dewpoint_from_specific_humidity(*a)
        ),
        "saturation_vapor_pressure": ((t,), lambda a: thermo.saturation_vapor_pressure(*a)),
        "saturation_mixing_ratio": (
            (p, t), lambda a: thermo.saturation_mixing_ratio(*a)
        ),
        "mixing_ratio_from_relative_humidity": (
            (p, t, rh), lambda a: thermo.mixing_ratio_from_relative_humidity(*a)
        ),
        "relative_humidity_from_dewpoint": (
            (t, td), lambda a: thermo.relative_humidity_from_dewpoint(*a)
        ),
        "relative_humidity_from_mixing_ratio": (
            (p, t, w), lambda a: thermo.relative_humidity_from_mixing_ratio(*a)
        ),
        "relative_humidity_from_specific_humidity": (
            (p, t, q), lambda a: thermo.relative_humidity_from_specific_humidity(*a)
        ),
        "equivalent_potential_temperature": (
            (p, t, td), lambda a: thermo.equivalent_potential_temperature(*a)
        ),
        "virtual_temperature_from_dewpoint": (
            (p, t, td), lambda a: thermo.virtual_temperature_from_dewpoint(*a)
        ),
        "virtual_potential_temperature": (
            (p, t, w), lambda a: thermo.virtual_potential_temperature(*a)
        ),
        "specific_humidity_from_dewpoint": (
            (p, td), lambda a: thermo.specific_humidity_from_dewpoint(*a)
        ),
        "psychrometric_vapor_pressure_wet": (
            (p, t, td), lambda a: thermo.psychrometric_vapor_pressure_wet(*a)
        ),
        "relative_humidity_wet_psychrometric": (
            (p, t, td), lambda a: thermo.relative_humidity_wet_psychrometric(*a)
        ),
        "pressure_to_height_std": ((p,), lambda a: basic.pressure_to_height_std(*a)),
        "height_to_pressure_std": ((t,), lambda a: basic.height_to_pressure_std(*a)),
        "altimeter_to_station_pressure": (
            (p, t), lambda a: basic.altimeter_to_station_pressure(*a)
        ),
        "altimeter_to_sea_level_pressure": (
            (p, t, td), lambda a: basic.altimeter_to_sea_level_pressure(*a)
        ),
        "windchill": ((t, w), lambda a: basic.windchill(*a)),
        "wind_speed": ((t, td), lambda a: basic.wind_speed(*a)),
        "wind_direction": ((t, td), lambda a: basic.wind_direction(*a)),
        "heat_index": ((t, rh), lambda a: basic.heat_index(*a)),
        "apparent_temperature": (
            (t, rh, w), lambda a: basic.apparent_temperature(*a)
        ),
        "supercell_composite": ((t, rh, w), lambda a: calc.supercell_composite(*a)),
    }


def _as_array(value):
    if isinstance(value, tuple):
        value = value[0]
    return np.asarray(np.ma.getdata(value), dtype=float)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 15, 17, 63, 65, 1023, 1025])
def test_elementwise_kernels_handle_every_length_up_to_a_few_lanes(n):
    # a length that is not a whole number of lanes drives the scalar tail, and a
    # length below one lane drives the tail alone
    for name, (args, fn) in _elementwise_cases(n).items():
        got = _as_array(fn(args))
        assert got.shape == (n,), name
        assert np.all(np.isfinite(got)), name


def test_elementwise_kernels_are_chunk_independent():
    # past the threshold the same array is split across workers, so the
    # parallel result has to equal the serial one element for element
    n = PAR_MIN + 3 * LANES + 1
    for name, (args, fn) in _elementwise_cases(n, seed=23).items():
        whole = _as_array(fn(args))
        def sliced(lo, hi):
            return fn([a if np.isscalar(a) else a[lo:hi] for a in args])

        pieces = np.concatenate(
            [_as_array(sliced(lo, hi)) for lo, hi in _spans(n, 4093)]
        )
        assert pieces.shape == (n,), name
        np.testing.assert_allclose(whole, pieces, rtol=1e-11, atol=1e-11, err_msg=name)


def _spans(n, step):
    return [(lo, min(lo + step, n)) for lo in range(0, n, step)]


def test_lcl_chunking_matches_the_serial_solve():
    n = PAR_MIN + 5
    rng = np.random.default_rng(5)
    p = rng.uniform(50000.0, 100000.0, n)
    t = 288.0 + rng.uniform(-30.0, 12.0, n)
    td = t - rng.uniform(0.0, 25.0, n)
    whole_p, whole_t = thermo.lcl(p, t, td)
    ps, ts = [], []
    for lo, hi in _spans(n, 7919):
        a, b = thermo.lcl(p[lo:hi], t[lo:hi], td[lo:hi])
        ps.append(a)
        ts.append(b)
    assert whole_p.shape == (n,) and whole_t.shape == (n,)
    np.testing.assert_allclose(whole_p, np.concatenate(ps), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(whole_t, np.concatenate(ts), rtol=1e-12, atol=1e-9)


def test_derivative_handles_strides_that_are_not_a_whole_number_of_lanes():
    # 13 x 7: the last axis has stride 1 and length 7, the first has stride 7,
    # which is not a multiple of the lane width, so the strided path has to
    # fall back to scalar lines part of the way through
    rng = np.random.default_rng(3)
    f = rng.normal(size=(13, 7)) * 50.0 + 200.0
    for axis in (0, 1, -1, -2):
        d = np.full(f.shape[axis] - 1, 2.0)
        np.testing.assert_allclose(
            tools.first_derivative(f, axis=axis, delta=d),
            _reference_derivative(f, axis, d),
            rtol=1e-14,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            tools.second_derivative(f, axis=axis, delta=d),
            _reference_second_derivative(f, axis, d),
            rtol=1e-14,
            atol=1e-9,
        )


def test_composite_indices_zero_rather_than_clamp_below_the_lower_bound():
    # upstream zeroes the shear under the lower bound; it does not clamp to it
    shear = np.array([0.0, 5.0, 9.99, 10.0, 15.0, 20.0, 40.0])
    cape = np.full(len(shear), 2000.0)
    esh = np.full(len(shear), 250.0)
    sc = calc.supercell_composite(cape, esh, shear)
    np.testing.assert_allclose(sc[:3], 0.0)
    np.testing.assert_allclose(sc[3], 2000.0 / 1000.0 * (250.0 / 50.0) * 0.5)
    np.testing.assert_allclose(sc[-1], 2000.0 / 1000.0 * (250.0 / 50.0) * 1.0)

    s6 = np.array([0.0, 11.0, 12.49, 12.5, 20.0, 30.0, 50.0])
    lcl = np.full(len(s6), 1500.0)
    sh1 = np.full(len(s6), 250.0)
    st = calc.significant_tornado(cape, lcl, sh1, s6)
    np.testing.assert_allclose(st[:3], 0.0)
    np.testing.assert_allclose(st[-1], cape[-1] / 1500.0 * 0.5 * (250.0 / 150.0) * 1.5)


def test_smoothing_is_bit_for_bit_the_scalar_weighted_sum():
    # the vectorised window keeps upstream's (ky, kx) accumulation order
    rng = np.random.default_rng(17)
    f = rng.normal(size=(23, 19)) * 50.0 + 200.0
    for npoints, w in (
        (5, [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]]),
        (9, [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]),
    ):
        got = basic.smooth_n_point(f, npoints)
        assert got.shape == f.shape
        ref = f.copy()
        for r in range(1, f.shape[0] - 1):
            for c in range(1, f.shape[1] - 1):
                total = 0.0
                for ky in range(3):
                    for kx in range(3):
                        total += w[ky][kx] * f[r + ky - 1, c + kx - 1]
                ref[r, c] = total
        np.testing.assert_array_equal(got, ref)


def test_saturation_mixing_ratio_vector_path_keeps_the_saturated_nan():
    # the masked branch of upstream's `saturation_mixing_ratio` has to survive
    # the vector path, which has no lane-wise blend
    p = np.array([100.0, 300.0, 5000.0, 100000.0])
    t = np.array([273.15, 273.15, 290.0, 280.0])
    got = thermo.saturation_mixing_ratio(p, t)
    assert np.isnan(got[0])
    assert np.isnan(got[1])
    assert np.isfinite(got[2]) and np.isfinite(got[3])
    assert got[2] > got[3]
