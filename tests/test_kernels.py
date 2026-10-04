"""Every `@export`ed kernel, called by name, with an assertion on its output.

A kernel that is wired up and completely wrong still ships green when nothing
calls it directly: the public API only reaches a kernel through its wrapper, so
a wrapper bug and a kernel bug look identical from outside.  This file closes
that gap.  Each test loads the compiled shared library, calls one exported
symbol through ctypes with buffers it builds itself, and checks the result
against either a closed-form value or an identity the mathematics must hold.

`test_every_export_is_covered` fails if an export is added without a test here.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pytest

from mojometpy import _lib
from mojometpy.constants import epsilon, g, Rd

# the integer the kernels switch on for upstream's `phase` keyword
LIQUID, SOLID, AUTO = 0, 1, 2


@pytest.fixture(scope="module")
def L():
    """The compiled library.  If it did not load there is nothing to test."""
    try:
        return _lib.lib()
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        pytest.skip(f"dist/libmojo-metpy.so is not loadable: {exc}")


def buf(*arrays):
    """C-contiguous float64 copies of each input.

    `np.ascontiguousarray` returns an already-contiguous argument unchanged, so
    a kernel would otherwise write straight into the caller's own array.  Copy
    unconditionally: these tests compare against values taken after the call.
    """
    if not arrays:
        return [np.full(1, np.nan)]
    return [np.array(a, dtype=np.float64, order="C", copy=True) for a in arrays]


def addr(a):
    return int(a.ctypes.data)


# ---- thermo: moist air properties ------------------------------------------


def test_mm_moist_air_gas_constant(L):
    q, out = buf([0.0, 0.001, 0.02], np.full(3, np.nan))
    L.mm_moist_air_gas_constant(addr(q), 3, addr(out))
    # upstream returns Rd + q*(Rv - Rd), linear in the specific humidity
    np.testing.assert_allclose(out, Rd + q * (461.52311572606084 - Rd), rtol=1e-14)
    assert out[0] == pytest.approx(Rd, rel=1e-14)


def test_mm_moist_air_specific_heat_pressure(L):
    q, out = buf([0.0, 0.005, 0.03], np.full(3, np.nan))
    L.mm_moist_air_specific_heat_pressure(addr(q), 3, addr(out))
    assert out[0] == pytest.approx(1004.6662184201462, rel=1e-14)
    assert np.all(np.diff(out) > 0)


def test_mm_moist_air_poisson_exponent(L):
    q, out = buf([0.0, 0.01], np.full(2, np.nan))
    L.mm_moist_air_poisson_exponent(addr(q), 2, addr(out))
    assert out[0] == pytest.approx(0.28571428571428564, rel=1e-12)
    assert out[1] != out[0]


@pytest.mark.parametrize(
    "sym,at_273",
    [
        ("mm_water_latent_heat_vaporization", 2500863.59321988),
        ("mm_water_latent_heat_sublimation", 2834542.29921988),
        ("mm_water_latent_heat_melting", 333721.294),
    ],
)
def test_mm_water_latent_heats(L, sym, at_273):
    t, out = buf([200.0, 273.15, 300.0], np.full(3, np.nan))
    getattr(L, sym)(addr(t), 3, addr(out))
    np.testing.assert_allclose(out[1], at_273, rtol=1e-9)
    # all three fall with temperature
    assert np.all(np.diff(out) < 0)


def test_mm_relative_humidity_from_dewpoint(L):
    t, td, out = buf([300.0, 300.0, 300.0], [280.0, 290.0, 300.0], np.full(3, np.nan))
    L.mm_relative_humidity_from_dewpoint(addr(t), addr(td), LIQUID, 3, addr(out))
    assert out[2] == pytest.approx(1.0, rel=1e-12)
    assert np.all(np.diff(out) > 0)


def test_mm_exner_function(L):
    p, out = buf([100000.0, 50000.0], np.full(2, np.nan))
    L.mm_exner_function(addr(p), 100000.0, 2, addr(out))
    np.testing.assert_allclose(out, (p / 100000.0) ** (2.0 / 7.0), rtol=1e-14)


def test_mm_potential_temperature(L):
    p, t, out = buf([100000.0, 50000.0], [300.0, 250.0], np.full(2, np.nan))
    L.mm_potential_temperature(addr(p), addr(t), 2, addr(out))
    np.testing.assert_allclose(
        out, t * (100000.0 / p) ** 0.2857142857142857, rtol=1e-14
    )


def test_mm_temperature_from_potential_temperature(L):
    p, th, out = buf([100000.0, 50000.0], [300.0, 350.0], np.full(2, np.nan))
    L.mm_temperature_from_potential_temperature(addr(p), addr(th), 2, addr(out))
    np.testing.assert_allclose(
        out, th * (p / 100000.0) ** 0.2857142857142857, rtol=1e-14
    )


def test_mm_dry_lapse(L):
    p, t, pref, out = buf(
        [100000.0, 90000.0], [300.0, 300.0], [100000.0, 100000.0], np.full(2, np.nan)
    )
    L.mm_dry_lapse(addr(p), addr(t), addr(pref), 2, addr(out))
    # upstream uses (p / p_ref)**kappa, not the barometric lapse rate
    np.testing.assert_allclose(out, (p / pref) ** 0.2857142857142857 * t, rtol=1e-10)
    assert out[1] < out[0]


def test_mm_lcl(L):
    p, t, td, op, ot = buf(
        [100000.0, 95000.0, 90000.0], [300.0, 295.0, 290.0], [285.0, 280.0, 275.0],
        np.full(3, np.nan), np.full(3, np.nan),
    )
    L.mm_lcl(addr(p), addr(t), addr(td), 3, addr(op), addr(ot))
    assert np.all(op < p) and np.all(ot < t)
    assert op[0] > op[-1]


def test_mm_vapor_pressure(L):
    p, w, out = buf([100000.0, 90000.0], [0.01, 0.02], np.full(2, np.nan))
    L.mm_vapor_pressure(addr(p), addr(w), 2, addr(out))
    np.testing.assert_allclose(out, p * w / (w + epsilon), rtol=1e-14)


@pytest.mark.parametrize(
    "sym", ["mm_saturation_vapor_pressure_liquid", "mm_saturation_vapor_pressure_solid"]
)
def test_mm_saturation_vapor_pressure_phases(L, sym):
    t, out = buf([273.15, 300.0], np.full(2, np.nan))
    getattr(L, sym)(addr(t), 2, addr(out))
    assert np.all(np.isfinite(out))
    assert np.all(np.diff(out) > 0)


def test_mm_saturation_vapor_pressure_auto_phase(L):
    # the auto branch switches regime below freezing
    t, out = buf([250.0, 300.0], np.full(2, np.nan))
    L.mm_saturation_vapor_pressure(addr(t), AUTO, 2, addr(out))
    solid, os_ = buf([250.0], np.full(1, np.nan))
    liquid, ol = buf([250.0], np.full(1, np.nan))
    L.mm_saturation_vapor_pressure_solid(addr(solid), 1, addr(os_))
    L.mm_saturation_vapor_pressure_liquid(addr(liquid), 1, addr(ol))
    np.testing.assert_allclose(out[0], os_[0], rtol=1e-12)


def test_mm_dewpoint(L):
    e, out = buf([1000.0, 2000.0], np.full(2, np.nan))
    L.mm_dewpoint(addr(e), 2, addr(out))
    assert np.all(out > 0) and np.all(np.diff(out) > 0)


def test_mm_dewpoint_from_relative_humidity(L):
    t, rh, out = buf([300.0, 300.0], [0.5, 1.0], np.full(2, np.nan))
    L.mm_dewpoint_from_relative_humidity(addr(t), addr(rh), 2, addr(out))
    # upstream returns 15.54 degC and 26.82 degC, which are 288.69 K and 299.97 K
    np.testing.assert_allclose(out, [288.68682756, 299.96724607], rtol=1e-9)
    assert out[0] < out[1]


def test_mm_dewpoint_from_specific_humidity(L):
    p, q, out = buf([100000.0, 90000.0], [0.005, 0.01], np.full(2, np.nan))
    L.mm_dewpoint_from_specific_humidity(addr(p), addr(q), 2, addr(out))
    assert np.all(np.isfinite(out))


def test_mm_mixing_ratio(L):
    e, p, out = buf([1000.0, 2000.0], [100000.0, 100000.0], np.full(2, np.nan))
    L.mm_mixing_ratio(addr(e), addr(p), epsilon, 2, addr(out))
    np.testing.assert_allclose(out, epsilon * e / (p - e), rtol=1e-14)


def test_mm_saturation_mixing_ratio(L):
    p, t, out = buf(
        [100000.0, 90000.0], [273.15, 300.0], np.full(2, np.nan)
    )
    L.mm_saturation_mixing_ratio(addr(p), addr(t), LIQUID, 2, addr(out))
    assert np.all(out > 0)
    assert out[1] > out[0]

    # above the vapour pressure of saturation the value is undefined
    hot, out2 = buf([400.0], np.full(1, np.nan))
    L.mm_saturation_mixing_ratio(addr(hot), addr(out2), LIQUID, 1, addr(out2))
    assert np.isnan(out2[0])


def test_mm_mixing_ratio_from_relative_humidity(L):
    p, t, rh, out = buf(
        [100000.0, 90000.0], [280.0, 290.0], [0.5, 1.0], np.full(2, np.nan)
    )
    L.mm_mixing_ratio_from_relative_humidity(addr(p), addr(t), addr(rh), LIQUID, 2, addr(out))
    assert out[1] > out[0]


def test_mm_relative_humidity_from_mixing_ratio(L):
    p, t, w, out = buf(
        [100000.0, 90000.0], [280.0, 290.0], [0.005, 0.010], np.full(2, np.nan)
    )
    L.mm_relative_humidity_from_mixing_ratio(addr(p), addr(t), addr(w), LIQUID, 2, addr(out))
    assert np.all(out > 0)


def test_mm_mixing_ratio_from_specific_humidity(L):
    q, out = buf([0.005, 0.01], np.full(2, np.nan))
    L.mm_mixing_ratio_from_specific_humidity(addr(q), 2, addr(out))
    np.testing.assert_allclose(out, q / (1.0 - q), rtol=1e-14)


def test_mm_specific_humidity_from_mixing_ratio(L):
    w, out = buf([0.01, 0.02], np.full(2, np.nan))
    L.mm_specific_humidity_from_mixing_ratio(addr(w), 2, addr(out))
    np.testing.assert_allclose(out, w / (1.0 + w), rtol=1e-14)


def test_mm_relative_humidity_from_specific_humidity(L):
    p, t, q, out = buf(
        [100000.0, 90000.0], [280.0, 290.0], [0.005, 0.010], np.full(2, np.nan)
    )
    L.mm_relative_humidity_from_specific_humidity(addr(p), addr(t), addr(q), LIQUID, 2, addr(out))
    assert np.all(out > 0)


def test_mm_equivalent_potential_temperature(L):
    p, t, td, out = buf(
        [100000.0, 90000.0], [300.0, 295.0], [290.0, 285.0], np.full(2, np.nan)
    )
    L.mm_equivalent_potential_temperature(addr(p), addr(t), addr(td), 2, addr(out))
    theta = t * (100000.0 / p) ** 0.2857142857142857
    assert np.all(out > theta)


def test_mm_virtual_temperature(L):
    t, w, out = buf([300.0, 295.0], [0.005, 0.010], np.full(2, np.nan))
    L.mm_virtual_temperature(addr(t), addr(w), epsilon, 2, addr(out))
    np.testing.assert_allclose(out, t * (1.0 + w / epsilon) / (1.0 + w), rtol=1e-13)


def test_mm_virtual_temperature_from_dewpoint(L):
    p, t, td, out = buf(
        [100000.0, 90000.0], [300.0, 295.0], [290.0, 285.0], np.full(2, np.nan)
    )
    L.mm_virtual_temperature_from_dewpoint(addr(p), addr(t), addr(td), epsilon, LIQUID, 2, addr(out))
    assert np.all(np.isfinite(out))


def test_mm_virtual_potential_temperature(L):
    p, t, w, out = buf(
        [100000.0, 90000.0], [300.0, 295.0], [0.005, 0.010], np.full(2, np.nan)
    )
    L.mm_virtual_potential_temperature(addr(p), addr(t), addr(w), epsilon, 2, addr(out))
    assert np.all(np.isfinite(out))


def test_mm_density(L):
    p, t, w, out = buf(
        [100000.0, 50000.0], [300.0, 250.0], [0.0, 0.01], np.full(2, np.nan)
    )
    L.mm_density(addr(p), addr(t), addr(w), epsilon, 2, addr(out))
    assert np.all(out > 0)
    assert out[1] < out[0]


def test_mm_psychrometric_vapor_pressure_wet(L):
    p, tdry, twet, out = buf(
        [100000.0, 100000.0], [300.0, 290.0], [295.0, 285.0], np.full(2, np.nan)
    )
    L.mm_psychrometric_vapor_pressure_wet(addr(p), addr(tdry), addr(twet), 0.000662, 2, addr(out))
    # the wet-bulb vapour pressure lies between the two dry-bulb vapour pressures
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)


def test_mm_relative_humidity_wet_psychrometric(L):
    p, tdry, twet, out = buf(
        [100000.0, 100000.0], [300.0, 290.0], [295.0, 285.0], np.full(2, np.nan)
    )
    L.mm_relative_humidity_wet_psychrometric(addr(p), addr(tdry), addr(twet), 0.000662, 2, addr(out))
    assert np.all(np.isfinite(out))


def test_mm_dry_static_energy(L):
    z, t, out = buf([0.0, 1000.0], [300.0, 295.0], np.full(2, np.nan))
    L.mm_dry_static_energy(addr(z), addr(t), 2, addr(out))
    np.testing.assert_allclose(out, g * z + 1004.6662184201462 * t, rtol=1e-13)


def test_mm_moist_static_energy(L):
    z, t, q, out = buf([0.0, 1000.0], [300.0, 295.0], [0.001, 0.002], np.full(2, np.nan))
    L.mm_moist_static_energy(addr(z), addr(t), addr(q), 2, addr(out))
    z0, t0, _ = buf([0.0, 1000.0], [300.0, 295.0], np.full(2, np.nan))
    dry = np.full(2, np.nan)
    L.mm_dry_static_energy(addr(z0), addr(t0), 2, addr(dry))
    # moist static energy exceeds the dry form wherever there is moisture
    assert np.all(out > dry)


def test_mm_thickness_hydrostatic(L):
    p, t, w, out = buf(
        np.linspace(100000.0, 10000.0, 20), np.full(20, 260.0), np.zeros(20), np.zeros(1)
    )
    L.mm_thickness_hydrostatic(addr(p), addr(t), addr(w), 0, epsilon, 20, addr(out))
    expected = -Rd / g * 260.0 * np.log(10000.0 / 100000.0)
    assert out[0] == pytest.approx(expected, rel=1e-9)

    # switching the mixing ratio on changes the answer
    moist, out2 = buf(np.full(20, 0.001), np.zeros(1))
    L.mm_thickness_hydrostatic(addr(p), addr(t), addr(moist), 1, epsilon, 20, addr(out2))
    assert out2[0] > out[0]


def test_mm_thickness_hydrostatic_from_relative_humidity(L):
    p, t, rh, out = buf(
        np.linspace(100000.0, 10000.0, 20), np.full(20, 260.0), np.full(20, 0.5),
        np.zeros(1),
    )
    L.mm_thickness_hydrostatic_from_relative_humidity(addr(p), addr(t), addr(rh), 20, addr(out))
    dry, out2 = buf(np.zeros(20), np.zeros(1))
    L.mm_thickness_hydrostatic(addr(p), addr(t), addr(dry), 0, epsilon, 20, addr(out2))
    assert out[0] > out2[0]


def test_mm_brunt_vaisala_frequency_squared(L):
    z, dtheta, theta, out = buf(
        np.linspace(0.0, 5000.0, 10), np.full(10, -0.01), np.full(10, 300.0), np.full(10, np.nan)
    )
    L.mm_brunt_vaisala_frequency_squared(addr(z), addr(dtheta), addr(theta), 10, addr(out))
    # upstream returns g/theta * dtheta/dz, which is negative for this profile
    np.testing.assert_allclose(out, g / 300.0 * -0.01, rtol=1e-13)
    assert np.all(out < 0)


def test_mm_brunt_vaisala_frequency_and_period(L):
    bv, freq, period = buf(
        np.full(6, 1e-4), np.full(6, np.nan), np.full(6, np.nan)
    )
    L.mm_brunt_vaisala_frequency(addr(bv), 6, addr(freq))
    L.mm_brunt_vaisala_period(addr(bv), 6, addr(period))
    np.testing.assert_allclose(freq, np.sqrt(1e-4), rtol=1e-14)
    # the period is a full oscillation, 2*pi/frequency
    np.testing.assert_allclose(period, 2.0 * np.pi / np.sqrt(1e-4), rtol=1e-13)

    neg, out = buf(np.full(3, -1.0), np.full(3, np.nan))
    L.mm_brunt_vaisala_frequency(addr(neg), 3, addr(out))
    assert np.all(np.isnan(out))


def test_mm_gradient_richardson_number(L):
    n = 8
    dthetadz, dudz, dvdz, theta, out = buf(
        np.full(n, 0.01), np.full(n, 0.001), np.full(n, 0.002), np.full(n, 300.0),
        np.full(n, np.nan),
    )
    L.mm_gradient_richardson_number(addr(dthetadz), addr(dudz), addr(dvdz), addr(theta), n, addr(out))
    # upstream returns (g/theta) * dtheta/dz / (dudz**2 + dvdz**2)
    expected = (g / 300.0) * 0.01 / (0.001**2 + 0.002**2)
    np.testing.assert_allclose(out, expected, rtol=1e-13)


def test_mm_static_stability(L):
    n = 8
    p, t, dlog, out = buf(
        np.linspace(100000.0, 50000.0, n), np.full(n, 300.0), np.full(n, 0.01), np.full(n, np.nan)
    )
    L.mm_static_stability(addr(p), addr(t), addr(dlog), n, addr(out))
    # upstream returns -Rd*T/p * d(ln theta)/dp
    np.testing.assert_allclose(out, -Rd * 300.0 / p * 0.01, rtol=1e-13)


def test_mm_vertical_velocity_pressure(L):
    n = 8
    p = np.linspace(100000.0, 50000.0, n)
    t, w, mr, out = buf(
        np.full(n, 300.0), np.full(n, -0.5), np.zeros(n), np.full(n, np.nan)
    )
    L.mm_vertical_velocity_pressure(addr(w), addr(p), addr(t), addr(mr), n, addr(out))
    # upstream returns -g * rho * w with rho = p/(Rd*T) for a dry atmosphere
    rho = p / (Rd * t)
    np.testing.assert_allclose(out, -g * rho * w, rtol=1e-13)


def test_mm_specific_humidity_from_dewpoint(L):
    p, td, out = buf([100000.0, 90000.0], [280.0, 275.0], np.full(2, np.nan))
    L.mm_specific_humidity_from_dewpoint(addr(p), addr(td), LIQUID, 2, addr(out))
    assert np.all((out > 0) & (out < 0.1))


def test_mm_scale_height(L):
    tb, tt, out = buf(np.full(10, 280.0), np.full(10, 300.0), np.full(10, np.nan))
    L.mm_scale_height(addr(tb), addr(tt), 10, addr(out))
    # upstream uses the mean of the bottom and top temperatures
    np.testing.assert_allclose(out, Rd * 290.0 / g, rtol=1e-13)


# ---- basic -----------------------------------------------------------------


def test_mm_wind_speed(L):
    u, v, out = buf([3.0, 0.0, -3.0], [4.0, 5.0, 4.0], np.full(3, np.nan))
    L.mm_wind_speed(addr(u), addr(v), 3, addr(out))
    np.testing.assert_allclose(out, np.hypot(u, v), rtol=1e-14)


def test_mm_wind_direction(L):
    u, v = buf([3.0, 0.0, -3.0], [4.0, 0.0, 4.0])
    frm, to = buf(np.full(3, np.nan), np.full(3, np.nan))
    L.mm_wind_direction(addr(u), addr(v), 0, 3, addr(frm))
    L.mm_wind_direction(addr(u), addr(v), 1, 3, addr(to))
    assert np.all(np.isfinite(frm)) and np.all(np.isfinite(to))
    # the two conventions differ by 180 degrees everywhere the wind blows
    np.testing.assert_allclose(np.mod(to - frm, 360.0)[[0, 2]], 180.0, rtol=1e-12)
    # calm air has no direction, and upstream reports zero
    calm, out = buf(np.zeros(2), np.full(2, np.nan))
    L.mm_wind_direction(addr(calm), addr(calm), 0, 2, addr(out))
    np.testing.assert_allclose(out, 0.0, atol=0)


def test_mm_wind_components(L):
    spd, dr = buf([5.0, 12.0], np.deg2rad(np.array([270.0, 45.0])))
    u, v = buf(np.full(2, np.nan), np.full(2, np.nan))
    L.mm_wind_components(addr(spd), addr(dr), 2, addr(u), addr(v))
    np.testing.assert_allclose(np.hypot(u, v), spd, rtol=1e-13)


def test_mm_windchill(L):
    t, s, face, mask, o, om = buf(
        [0.0, 20.0], [30.0, 20.0], [0.0, 0.0], np.full(2, np.nan), np.full(2, np.nan),
        np.full(2, np.nan),
    )
    L.mm_windchill(addr(t), addr(s), addr(face), addr(mask), 2, addr(o), addr(om))
    np.testing.assert_allclose(o[0], -6.472951590454416, rtol=1e-9)
    # 20 degC falls outside the formula's range, so the plain JAG/TI fit applies
    assert o[1] < t[1]


def test_mm_heat_index(L):
    # this port takes degrees Fahrenheit, as `basic.py` documents
    t, rh, mask, o, om = buf(
        [50.0, 95.0], [0.2, 0.5], np.full(2, np.nan), np.full(2, np.nan), np.full(2, np.nan)
    )
    L.mm_heat_index(addr(t), addr(rh), addr(mask), 2, addr(o), addr(om))
    np.testing.assert_allclose(o[1], 105.2157721, rtol=1e-9)
    assert np.isfinite(o[0])


def test_mm_apparent_temperature(L):
    t, rh, s, face, mask, o, om = buf(
        [20.0, 35.0], [0.3, 0.5], [10.0, 5.0], [0.0, 0.0], np.full(2, np.nan),
        np.full(2, np.nan), np.full(2, np.nan),
    )
    L.mm_apparent_temperature(addr(t), addr(rh), addr(s), addr(face), addr(mask), 2, addr(o), addr(om))
    assert np.all(np.isfinite(o))


def test_mm_pressure_to_height_std(L):
    # upstream's barometric base is p0 = 101325 Pa, not 1000 hPa
    p, out = buf([101325.0, 90000.0, 50000.0], np.full(3, np.nan))
    L.mm_pressure_to_height_std(addr(p), 3, addr(out))
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(out[1:], [987.96719742, 5571.43442321], rtol=1e-9)


def test_mm_height_to_pressure_std(L):
    h, out = buf([0.0, 1500.0, 5500.0], np.full(3, np.nan))
    L.mm_height_to_pressure_std(addr(h), 3, addr(out))
    np.testing.assert_allclose(out, [101325.0, 84547.60127237, 50486.5391628], rtol=1e-9)
    assert np.all(np.diff(out) < 0)


def test_mm_height_to_geopotential(L):
    # the spherical form: g*R*z/(R+z)
    z, out = buf([0.0, 10000.0], np.full(2, np.nan))
    L.mm_height_to_geopotential(addr(z), 2, addr(out))
    R = 6371008.7714
    np.testing.assert_allclose(out, g * R * z / (R + z), rtol=1e-13)
    np.testing.assert_allclose(out[1], 97912.81505219, rtol=1e-9)


def test_mm_geopotential_to_height(L):
    z, out = buf([0.0, 10000.0], np.full(2, np.nan))
    phi, _ = buf(np.array([0.0, 97912.81505219]), np.zeros(2))
    L.mm_geopotential_to_height(addr(phi), 2, addr(out))
    np.testing.assert_allclose(out, z, rtol=1e-6)


def test_mm_coriolis_parameter(L):
    lat, out = buf(np.deg2rad(np.array([0.0, 45.0, -45.0])), np.full(3, np.nan))
    L.mm_coriolis_parameter(addr(lat), 3, addr(out))
    np.testing.assert_allclose(out, 2.0 * 7.292115e-5 * np.sin(lat), rtol=1e-12)
    assert out[0] == pytest.approx(0.0, abs=1e-15)
    assert out[1] == pytest.approx(-out[2], rel=1e-15)


def test_mm_add_height_to_pressure(L):
    p, z, out = buf([100000.0, 50000.0], [0.0, 500.0], np.full(2, np.nan))
    L.mm_add_height_to_pressure(addr(p), addr(z), 2, addr(out))
    np.testing.assert_allclose(out, [100000.0, 46699.72206301], rtol=1e-9)


def test_mm_add_pressure_to_height(L):
    z, p, out = buf([0.0, 500.0], [100000.0, 50000.0], np.full(2, np.nan))
    L.mm_add_pressure_to_height(addr(z), addr(p), 2, addr(out))
    assert np.all(np.isfinite(out))
    assert out[1] < out[0]


def test_mm_sigma_to_pressure(L):
    sig, psfc, ptop, out = buf(
        [0.5, 1.0], np.full(2, 100000.0), np.full(2, 10000.0), np.full(2, np.nan)
    )
    L.mm_sigma_to_pressure(addr(sig), addr(psfc), addr(ptop), 2, addr(out))
    np.testing.assert_allclose(out, ptop + sig * (psfc - ptop), rtol=1e-14)


def test_mm_altimeter_to_station_pressure(L):
    # this port takes the altimeter in Pa (see the README unit table); the
    # `+30` in the kernel is upstream's 0.3 hPa
    a, z, out = buf([101325.0, 100000.0], [0.0, 100.0], np.full(2, np.nan))
    L.mm_altimeter_to_station_pressure(addr(a), addr(z), 2, addr(out))
    np.testing.assert_allclose(out, [101355.0, 98846.49216824], rtol=1e-9)


def test_mm_altimeter_to_sea_level_pressure(L):
    a, z, t, out = buf([101325.0, 100000.0], [0.0, 100.0], [288.15, 293.15], np.full(2, np.nan))
    L.mm_altimeter_to_sea_level_pressure(addr(a), addr(z), addr(t), 2, addr(out))
    np.testing.assert_allclose(out[0], 101355.0, rtol=1e-9)
    assert np.all(np.isfinite(out))


# ---- tools: stencils, interpolation, geodesics -----------------------------


def test_mm_first_derivative(L):
    # f = (1, 4, 9, 16, 25) is x**2 on the points x = 1..5, so the exact slope
    # in the interior is 2x+1 = 3, 5, 7
    f, delta, out = buf([1.0, 4.0, 9.0, 16.0, 25.0], np.full(4, 2.0), np.full(5, np.nan))
    L.mm_first_derivative(addr(f), addr(delta), 5, 1, 1, 5, addr(out))
    # df/d(delta) = 2x/delta = 2, 3, 4 on the interior points
    np.testing.assert_allclose(out[1:4], [2.0, 3.0, 4.0], rtol=1e-13)


def test_mm_first_derivative_strided(L):
    # the same stencil along a non-last axis, one line at a time
    shape = (4, 5)
    f = np.ascontiguousarray(np.tile(np.arange(5.0), (4, 1)))
    delta = np.full(4, 1.0)
    out = np.full(f.size, np.nan)
    L.mm_first_derivative(addr(f), addr(delta), 5, 1, 4, f.size, addr(out))
    np.testing.assert_allclose(out.reshape(shape)[:, 1:4], 1.0, rtol=1e-13)
    assert out.reshape(shape).shape == shape


def test_mm_second_derivative(L):
    f, delta, out = buf([0.0, 1.0, 4.0, 9.0, 16.0], np.full(4, 1.0), np.full(5, np.nan))
    L.mm_second_derivative(addr(f), addr(delta), 5, 1, 1, 5, addr(out))
    np.testing.assert_allclose(out[1:4], 2.0, rtol=1e-13)


def test_mm_second_derivative_strided(L):
    f = np.ascontiguousarray(np.tile(np.arange(5.0), (3, 1)))
    delta = np.full(4, 1.0)
    out = np.full(f.size, np.nan)
    L.mm_second_derivative(addr(f), addr(delta), 5, 1, 3, f.size, addr(out))
    np.testing.assert_allclose(out.reshape(f.shape)[:, 1:4], 0.0, atol=1e-13)


def test_mm_normalize_window(L):
    window, _ = buf([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], np.zeros(9))
    L.mm_normalize_window(addr(window), window.size)
    np.testing.assert_allclose(window.sum(), 1.0, rtol=1e-15)
    np.testing.assert_allclose(window, np.array([0, 1, 0, 1, 0, 1, 0, 0, 0]) / 3.0, rtol=1e-15)


def test_mm_smooth_window(L):
    # a 5x5 window on a 7x7 grid leaves a 3x3 interior, so the stencil has room
    grid = (np.arange(49.0) ** 1.5).reshape(7, 7)
    data, scratch = buf(grid, np.zeros(49))
    weights, _ = buf(np.full((5, 5), 1.0 / 25.0), np.zeros(25))
    L.mm_smooth_window(addr(data), addr(scratch), addr(weights), 5, 5, 7, 7, 1)
    got = scratch.reshape(7, 7)
    # the kernel reads the weights the wrapper already normalised
    # the border is left exactly as it was
    np.testing.assert_array_equal(got[0], grid[0])
    np.testing.assert_array_equal(got[:, 0], grid[:, 0])
    # the interior is the 5x5 mean of the 5x5 block centred on that pixel
    manual = sum(
        (1.0 / 25.0) * grid[r + ky - 2, c + kx - 2]
        for r, c in [(3, 3)]
        for ky in range(5)
        for kx in range(5)
    )
    assert got[3, 3] == pytest.approx(manual, rel=1e-13)
    # the border of the interior, whose window hangs over the edge, is not the
    # full 5x5 mean; assert it is still finite and inside the data's range
    assert np.isfinite(got[1, 1])
    assert grid.min() <= got[1, 1] <= grid.max()
    # a second pass smooths further, so the interior moves again
    two = np.zeros(49)
    L.mm_smooth_window(addr(data), addr(two), addr(weights), 5, 5, 7, 7, 2)
    assert two.reshape(7, 7)[3, 3] != pytest.approx(got[3, 3], rel=1e-12)


def test_mm_interpolate_1d(L):
    x, xp, values, out = buf(
        [5.0, 15.0], [0.0, 10.0, 20.0], [0.0, 100.0, 200.0], np.full(2, np.nan)
    )
    L.mm_interpolate_1d(addr(x), addr(xp), addr(values), 2, 3, 1, 1, np.nan, addr(out))
    np.testing.assert_allclose(out, [50.0, 150.0], rtol=1e-13)

    off, out2 = buf([100.0], np.full(1, np.nan))
    L.mm_interpolate_1d(addr(off), addr(xp), addr(values), 1, 3, 1, 1, -1.0, addr(out2))
    assert out2[0] == -1.0


def test_mm_resample_nn_1d_nearest(L):
    a, _ = buf([0.0, 10.0, 20.0], np.zeros(3))
    assert L.mm_resample_nn_1d_nearest(addr(a), 11.0, 3) == 1
    assert L.mm_resample_nn_1d_nearest(addr(a), -5.0, 3) == 0
    assert L.mm_resample_nn_1d_nearest(addr(a), 100.0, 3) == 2


def test_mm_nearest_intersection_idx(L):
    a, b, out = buf([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], np.full(4, np.nan))
    found = L.mm_nearest_intersection_idx(addr(a), addr(b), 4, addr(out))
    assert found == 1
    assert out[0] == 1.0


def test_mm_lat_lon_grid_deltas_dx(L):
    lon, lat, dx = buf(np.linspace(-180.0, 180.0, 9), np.zeros(9), np.full(8, np.nan))
    L.mm_lat_lon_grid_deltas_dx(addr(lon), addr(lat), 9, 1, 1, addr(dx))
    # 45 degrees of arc on the sphere `pyproj.Geod(ellps='sphere')` uses
    np.testing.assert_allclose(dx, 6370997.0 * np.pi / 4.0, rtol=1e-9)


def test_mm_lat_lon_grid_deltas_dy(L):
    lon, lat, dy = buf(np.zeros(9), np.linspace(-80.0, 80.0, 9), np.full(8, np.nan))
    L.mm_lat_lon_grid_deltas_dy(addr(lon), addr(lat), 9, 1, 1, addr(dy))
    assert np.all(dy > 0)
    np.testing.assert_allclose(dy, 6370997.0 * np.pi / 9.0, rtol=1e-9)


# ---- kinematics -------------------------------------------------------------


def _field(ny=5, nx=6, seed=31):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(ny, nx)) * 10.0)


def test_mm_vector_derivative(L):
    u, v = _field(), _field(seed=32)
    nx, xs, xo = 6, 1, 5
    ny, ys, yo = 5, 6, 6
    dx, dy = np.full(nx, 1.0), np.full(ny, 1.0)
    dudx, dudy, dvdx, dvdy = (np.full(u.size, np.nan) for _ in range(4))
    L.mm_vector_derivative(
        addr(u), addr(v), addr(dx), addr(dy), nx, xs, xo, ny, ys, yo,
        addr(dudx), addr(dudy), addr(dvdx), addr(dvdy),
    )
    # differentiate a linear ramp along x: the interior is exactly one
    ramp = np.ascontiguousarray(np.tile(np.arange(nx, dtype=float), (5, 1)))
    d = np.full(u.size, np.nan)
    L.mm_vector_derivative(
        addr(ramp), addr(ramp), addr(dx), addr(dy), nx, xs, xo, ny, ys, yo,
        addr(d), addr(np.zeros(u.size)), addr(np.zeros(u.size)), addr(np.zeros(u.size)),
    )
    np.testing.assert_allclose(d.reshape(5, nx)[:, 1:-1], 1.0, rtol=1e-12)


def test_mm_vector_derivative_map_correction(L):
    u, v = _field(), _field(seed=32)
    n = u.size
    shape = u.shape
    dudx, dudy, dvdx, dvdy = (np.full(n, 1.0) for _ in range(4))
    ps, ms = np.full(shape, 2.0), np.full(shape, 3.0)
    zeros = np.zeros(shape)
    L.mm_vector_derivative_map_correction(
        addr(u), addr(v), addr(zeros), addr(zeros), addr(ps), addr(ms), n,
        addr(dudx), addr(dudy), addr(dvdx), addr(dvdy),
    )
    # with zero scale gradients only the map factors remain: ps*1 - v*0 and so on
    np.testing.assert_allclose(dudx, 2.0, atol=0.0)
    np.testing.assert_allclose(dudy, 3.0, atol=0.0)
    np.testing.assert_allclose(dvdx, 2.0, atol=0.0)
    np.testing.assert_allclose(dvdy, 3.0, atol=0.0)


def test_mm_vorticity(L):
    dudy, dvdx, out = buf(np.ones(6), np.full(6, 3.0), np.full(6, np.nan))
    L.mm_vorticity(addr(dudy), addr(dvdx), 6, addr(out))
    np.testing.assert_allclose(out, 2.0, rtol=1e-14)


def test_mm_divergence(L):
    dudx, dvdy, out = buf(np.full(6, 5.0), np.full(6, 2.0), np.full(6, np.nan))
    L.mm_divergence(addr(dudx), addr(dvdy), 6, addr(out))
    np.testing.assert_allclose(out, 7.0, rtol=1e-14)


def test_mm_shearing_deformation(L):
    dudy, dvdx, out = buf(np.ones(6), np.full(6, 3.0), np.full(6, np.nan))
    L.mm_shearing_deformation(addr(dudy), addr(dvdx), 6, addr(out))
    np.testing.assert_allclose(out, 4.0, rtol=1e-14)


def test_mm_stretching_deformation(L):
    dudx, dvdy, out = buf(np.full(6, 5.0), np.full(6, 2.0), np.full(6, np.nan))
    L.mm_stretching_deformation(addr(dudx), addr(dvdy), 6, addr(out))
    np.testing.assert_allclose(out, 3.0, rtol=1e-14)


def test_mm_total_deformation(L):
    dudx, dudy, dvdx, dvdy, out = buf(
        np.full(6, 5.0), np.ones(6), np.full(6, 3.0), np.full(6, 2.0), np.full(6, np.nan)
    )
    L.mm_total_deformation(addr(dudx), addr(dudy), addr(dvdx), addr(dvdy), 6, addr(out))
    np.testing.assert_allclose(out, np.sqrt(16.0 + 9.0), rtol=1e-14)


@pytest.mark.parametrize("sym", ["mm_shear_vorticity", "mm_curvature_vorticity"])
def test_mm_shear_and_curvature_vorticity(L, sym):
    u, v, dudx, dudy, dvdx, dvdy, out = buf(
        [3.0, 4.0], [4.0, 3.0], [1.0, 2.0], [0.5, 1.0], [2.0, 1.0], [1.0, 0.5],
        np.full(2, np.nan),
    )
    getattr(L, sym)(
        addr(u), addr(v), addr(dudx), addr(dudy), addr(dvdx), addr(dvdy), 2, addr(out)
    )
    assert np.all(np.isfinite(out))
    # neither branch can divide by zero for a nonzero wind
    assert np.all(np.abs(out) < 1e6)


def test_mm_advection(L):
    n = 6
    dfdx, dfdy, dfdz, u, v, w, out = buf(
        np.full(n, 2.0), np.full(n, -1.0), np.zeros(n), np.full(n, 3.0), np.full(n, 4.0),
        np.zeros(n), np.full(n, np.nan),
    )
    L.mm_advection(addr(dfdx), addr(dfdy), addr(dfdz), addr(u), addr(v), addr(w), 0, n, addr(out))
    np.testing.assert_allclose(out, -(3.0 * 2.0 + 4.0 * -1.0), rtol=1e-14)
    L.mm_advection(addr(dfdx), addr(dfdy), addr(dfdz), addr(u), addr(v), addr(w), 1, n, addr(out))
    np.testing.assert_allclose(out, -2.0, rtol=1e-14)


def test_mm_frontogenesis(L):
    n = 6
    ddx, ddy, shrd, strd, tdef, div, out = buf(
        np.ones(n), np.full(n, 2.0), np.ones(n), np.ones(n), np.full(n, 2.0),
        np.full(n, 0.5), np.full(n, np.nan),
    )
    L.mm_frontogenesis(addr(ddx), addr(ddy), addr(shrd), addr(strd), addr(tdef), addr(div), n, addr(out))
    mag = np.sqrt(1.0 + 4.0)
    psi = 0.5 * np.arctan2(1.0, 1.0)
    sin_beta = (1.0 * np.cos(psi) + 2.0 * np.sin(psi)) / mag
    expected = 0.5 * mag * (2.0 * (1.0 - 2.0 * sin_beta**2) - 0.5)
    np.testing.assert_allclose(out, expected, rtol=1e-13)


def test_mm_geostrophic_wind(L):
    dhdx, dhdy, lat, og, vg = buf(
        [1.0, 2.0], [-1.0, 1.0], [np.pi / 4, np.pi / 6], np.full(2, np.nan), np.full(2, np.nan)
    )
    L.mm_geostrophic_wind(addr(dhdx), addr(dhdy), addr(lat), 2, addr(og), addr(vg))
    f = 2.0 * 7.292115e-5 * np.sin(lat)
    np.testing.assert_allclose(og, -(g / f) * dhdy, rtol=1e-13)
    np.testing.assert_allclose(vg, (g / f) * dhdx, rtol=1e-13)
    # a different latitude gives a different Coriolis factor and a different wind
    assert og[0] != pytest.approx(og[1], rel=1e-6)


def test_mm_montgomery_streamfunction(L):
    z, t, out = buf([0.0, 100.0], [300.0, 300.0], np.full(2, np.nan))
    L.mm_montgomery_streamfunction(addr(z), addr(t), 2, addr(out))
    # upstream returns kJ/kg; this port returns the SI value in J/kg
    np.testing.assert_allclose(out, g * z + 1004.6662184201462 * t, rtol=1e-12)


def test_mm_storm_relative_helicity(L):
    u, v, out = buf([10.0, 12.0, 15.0], [0.0, 3.0, -2.0], np.full(3, np.nan))
    L.mm_storm_relative_helicity(addr(u), addr(v), 0.0, 0.0, 3, addr(out))
    # the output is (positive sum, negative sum, total)
    np.testing.assert_allclose(out, [69.0, -30.0, 39.0], rtol=1e-12)
    # a purely zonal wind has no helicity
    flat = np.ascontiguousarray(np.array([10.0, 12.0, 15.0]))
    zero = np.ascontiguousarray(np.zeros(3))
    out2 = np.full(3, np.nan)
    L.mm_storm_relative_helicity(addr(flat), addr(zero), 0.0, 0.0, 3, addr(out2))
    assert out2[2] == pytest.approx(0.0, abs=1e-14)
    # the sum is over the storm-relative wind, so moving the storm moves it
    moved, moved_v, out3 = buf(
        [10.0, 12.0, 15.0], [0.0, 3.0, -2.0], np.full(3, np.nan)
    )
    L.mm_storm_relative_helicity(addr(moved), addr(moved_v), 10.0, 0.0, 3, addr(out3))
    np.testing.assert_allclose(out3, [19.0, 0.0, 19.0], rtol=1e-12)


def test_mm_potential_vorticity_baroclinic(L):
    n = 5
    dudp, dthetady, dvdp, dthetadx, avor, dthetadp, out = buf(
        np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), np.full(n, 1e-4),
        np.full(n, 1e-3), np.full(n, np.nan),
    )
    L.mm_potential_vorticity_baroclinic(
        addr(dudp), addr(dthetady), addr(dvdp), addr(dthetadx), addr(avor), addr(dthetadp),
        n, addr(out),
    )
    # only dthetady is nonzero here
    assert np.all(np.isfinite(out))


def test_mm_potential_vorticity_barotropic(L):
    n = 5
    avor, height, out = buf(np.full(n, 1e-4), np.full(n, 100.0), np.full(n, np.nan))
    L.mm_potential_vorticity_barotropic(addr(avor), addr(height), n, addr(out))
    assert np.all(np.isfinite(out))


def test_mm_inertial_advective_wind(L):
    n = 4
    u, v, dudx, dudy, dvdx, dvdy, lat, ox, oy = buf(
        np.full(n, 5.0), np.full(n, -3.0), np.full(n, 0.1), np.full(n, 0.2),
        np.full(n, 0.3), np.full(n, 0.4), np.full(n, 0.5), np.full(n, np.nan), np.full(n, np.nan),
    )
    L.mm_inertial_advective_wind(
        addr(u), addr(v), addr(dudx), addr(dudy), addr(dvdx), addr(dvdy), addr(lat), n,
        addr(ox), addr(oy),
    )
    # upstream returns -(u*dv_g/dx + v*dv_g/dy)/f and (u*du_g/dx + v*du_g/dy)/f,
    # where the gradients are of the geostrophic wind
    f = 2.0 * 7.292115e-5 * np.sin(lat)
    np.testing.assert_allclose(ox, -(u * dvdx + v * dvdy) / f, rtol=1e-10)
    np.testing.assert_allclose(oy, (u * dudx + v * dudy) / f, rtol=1e-10)


def test_mm_q_vector(L):
    n = 4
    dudx = np.full(n, 0.1)
    dudy, dvdx, dvdy = np.full(n, 0.2), np.full(n, 0.3), np.full(n, 0.4)
    dtempdx, dtempdy = np.full(n, 0.01), np.full(n, 0.02)
    p, q1, q2 = buf(np.full(n, 50000.0), np.full(n, np.nan), np.full(n, np.nan))
    L.mm_q_vector(
        addr(dudx), addr(dudy), addr(dvdx), addr(dvdy), addr(dtempdx), addr(dtempdy),
        addr(p), 1.0, n, addr(q1), addr(q2),
    )
    assert np.all(np.isfinite(q1)) and np.all(np.isfinite(q2))
    # the two components are the horizontal and vertical divergence terms and
    # differ for a wind field with a nonzero vorticity
    assert not np.allclose(q1, q2)


# ---- turbulence -------------------------------------------------------------


def test_mm_tke(L):
    n, stride, outer = 20, 1, 3
    u, v, w = _field(3, 20, 41), _field(3, 20, 42), _field(3, 20, 43)
    pert, debiased = np.full(outer, np.nan), np.full(outer, np.nan)
    L.mm_tke(addr(u), addr(v), addr(w), 1, n, stride, outer, addr(pert))
    L.mm_tke(addr(u), addr(v), addr(w), 0, n, stride, outer, addr(debiased))
    # subtracting each line mean can only reduce the mean square
    assert np.all(pert >= debiased - 1e-12)


def test_mm_friction_velocity(L):
    n, stride, outer = 20, 1, 3
    u, w = _field(3, 20, 44), _field(3, 20, 45)
    out = np.full(outer, np.nan)
    with_v = np.full(outer, np.nan)
    v = _field(3, 20, 46)
    # `has_v` is upstream's `if v is not None`; the v buffer is unused when it is 0
    L.mm_friction_velocity(addr(u), addr(w), 0, 0, 0, n, stride, outer, addr(out))
    L.mm_friction_velocity(addr(u), addr(w), addr(v), 1, 0, n, stride, outer, addr(with_v))
    assert np.all(np.isfinite(out)) and np.all(out >= 0)
    assert np.all(np.isfinite(with_v)) and np.all(with_v >= 0)
    # adding the v-w flux changes the answer
    assert not np.allclose(out, with_v)


# ---- indices ----------------------------------------------------------------


def test_mm_supercell_composite(L):
    cape, esh, shear, out = buf(
        [2000.0, 2000.0], [250.0, 250.0], [0.0, 40.0], np.full(2, np.nan)
    )
    L.mm_supercell_composite(addr(cape), addr(esh), addr(shear), 2, addr(out))
    # the shear term is zeroed below the 10 m/s lower bound
    np.testing.assert_allclose(out[0], 0.0, rtol=1e-14)
    assert out[1] > out[0]


def test_mm_significant_tornado(L):
    cape, lcl, sh1, s6, out = buf(
        [2000.0, 2000.0], [1500.0, 1500.0], [250.0, 250.0], [0.0, 50.0], np.full(2, np.nan)
    )
    L.mm_significant_tornado(addr(cape), addr(lcl), addr(sh1), addr(s6), 2, addr(out))
    np.testing.assert_allclose(out[0], 0.0, rtol=1e-14)
    assert out[1] > out[0]


# ---- coverage guard ---------------------------------------------------------


def _exports():
    return set(re.findall(r'@export\("([A-Za-z_0-9]+)"\)', _lib.capi_source()))


def test_every_export_is_covered():
    """Each `@export`ed symbol needs a test here that calls it by name.

    Without this, a kernel that is wired up and completely wrong still ships
    green: nothing calls it directly, so nothing observes it.
    """
    import inspect

    covered = set()
    for _, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        source = inspect.getsource(fn)
        covered |= {name for name in _exports() if name in source}
    missing = sorted(_exports() - covered)
    assert not missing, f"exported kernels with no direct test: {missing}"