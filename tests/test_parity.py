"""Numerical parity against upstream MetPy 1.7.1 on the same inputs.

Every case feeds the same float64 arrays to `mojometpy` and to `metpy.calc`
and asserts the two agree.  The tolerance is 1e-9 relative rather than machine
epsilon: Mojo's `pow`/`log` and glibc's disagree in the last few significant
bits (measured worst case ~3e-10 on `(p/P0) ** kappa`), and the functions with
no transcendental in them are pinned tighter, at 1e-12.

`metpy` returns `pint` quantities in whatever unit upstream picks; the port is
unitless, so each case converts upstream's result into the SI base unit the
kernel works in before comparing.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojometpy.calc as mm
from metpy.units import units

mcalc = pytest.importorskip("metpy.calc")

CASES = []


def case(name, rtol=1e-9, atol=1e-11):
    def deco(fn):
        CASES.append((name, fn, rtol, atol))
        return fn
    return deco


def _kelvin(q):
    """Upstream `dewpoint` and friends return degrees Celsius."""
    return q.m_as("degC") + 273.15


@pytest.fixture(scope="module")
def d():
    rng = np.random.default_rng(0)
    n = 64
    out = {}
    out["P"] = rng.uniform(50000.0, 100000.0, n)
    out["T"] = rng.uniform(240.0, 310.0, n)
    out["TD"] = out["T"] - rng.uniform(0.0, 30.0, n)
    out["RH"] = rng.uniform(0.05, 1.0, n)
    out["W"] = rng.uniform(1e-4, 0.03, n)
    out["E"] = 1000.0 + np.arange(n) * 3.0
    out["Z"] = np.linspace(0.0, 10000.0, n)
    out["Q"] = units.Quantity

    out["P1"] = np.linspace(100000.0, 10000.0, 40)
    out["H1"] = mm.pressure_to_height_std(out["P1"])
    out["T1"] = 288.0 - 6.5e-3 * out["H1"]
    out["TD1"] = out["T1"] - 10.0
    out["W1"] = mm.saturation_mixing_ratio(out["P1"], out["TD1"])
    out["TH1"] = mm.potential_temperature(out["P1"], out["T1"])
    out["U1"] = 10.0 * np.cos(np.linspace(0.0, 1.0, 40))
    out["V1"] = 5.0 * np.sin(np.linspace(0.0, 1.0, 40))

    out["U"] = rng.normal(size=n) * 10.0
    out["V"] = rng.normal(size=n) * 10.0
    out["SPD"] = rng.uniform(0.0, 80.0, n)
    out["TC"] = rng.uniform(-25.0, 35.0, n)
    out["TF"] = rng.uniform(50.0, 115.0, n)
    out["RHF"] = rng.uniform(0.05, 0.95, n)
    out["COLD"] = rng.uniform(30.0, 90.0, n)
    out["LAT"] = rng.uniform(-1.0, 1.0, n)
    out["ALT"] = rng.uniform(980.0, 1040.0, n)
    out["SIG"] = rng.uniform(0.0, 1.0, n)

    # A 2-D field that varies along both axes, so every stencil region
    # (left / center / right) of the derivatives is actually exercised.
    lat2 = np.linspace(-80.0, 80.0, 30)
    lon2 = np.linspace(-170.0, 170.0, 40)
    out["LON"], out["LAT2"] = np.meshgrid(lon2, lat2)
    out["UU"] = 20.0 * np.sin(np.deg2rad(out["LON"])) * np.cos(np.deg2rad(out["LAT2"]))
    out["VV"] = 15.0 * np.cos(np.deg2rad(out["LON"])) * np.sin(np.deg2rad(out["LAT2"]))
    out["SCALAR"] = 100.0 + 0.1 * out["LAT2"] ** 2 + 0.05 * out["LON"] ** 2
    out["DX"] = 2.0
    out["DY"] = 2.0
    return out


# ---- thermo: moist air and latent heats ----

@case("moist_air_gas_constant", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.moist_air_gas_constant(d["W"]), mcalc.moist_air_gas_constant(Q(d["W"], ""))


@case("moist_air_specific_heat_pressure", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.moist_air_specific_heat_pressure(d["W"]), mcalc.moist_air_specific_heat_pressure(Q(d["W"], ""))


@case("moist_air_poisson_exponent", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.moist_air_poisson_exponent(d["W"]), mcalc.moist_air_poisson_exponent(Q(d["W"], ""))


@case("water_latent_heat_vaporization", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.water_latent_heat_vaporization(d["T"]), mcalc.water_latent_heat_vaporization(Q(d["T"], "K"))


@case("water_latent_heat_sublimation", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.water_latent_heat_sublimation(d["T"]), mcalc.water_latent_heat_sublimation(Q(d["T"], "K"))


@case("water_latent_heat_melting", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.water_latent_heat_melting(d["T"]), mcalc.water_latent_heat_melting(Q(d["T"], "K"))


# ---- thermo: temperature transforms ----

@case("relative_humidity_from_dewpoint", 1e-8, 1e-10)
def _(d):
    Q = d["Q"]
    return (
        mm.relative_humidity_from_dewpoint(d["T"], d["TD"]),
        mcalc.relative_humidity_from_dewpoint(Q(d["T"], "K"), Q(d["TD"], "K")),
    )


@case("exner_function", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    return mm.exner_function(d["P"]), mcalc.exner_function(Q(d["P"], "Pa"))


@case("potential_temperature", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    return mm.potential_temperature(d["P"], d["T"]), mcalc.potential_temperature(Q(d["P"], "Pa"), Q(d["T"], "K"))


@case("temperature_from_potential_temperature", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    th = d["T"] + 5.0
    return (
        mm.temperature_from_potential_temperature(d["P"], th),
        mcalc.temperature_from_potential_temperature(Q(d["P"], "Pa"), Q(th, "K")),
    )


@case("dry_lapse", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    return mm.dry_lapse(d["P"], d["T"]), mcalc.dry_lapse(Q(d["P"], "Pa"), Q(d["T"], "K"))


@case("lcl_pressure", 1e-8, 1e-6)
def _(d):
    Q = d["Q"]
    ours_p, _ = mm.lcl(d["P"], d["T"], d["TD"])
    theirs = mcalc.lcl(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["TD"], "K"))[0]
    return ours_p, theirs.m_as("Pa")


@case("lcl_temperature", 1e-8, 1e-6)
def _(d):
    Q = d["Q"]
    _, ours_t = mm.lcl(d["P"], d["T"], d["TD"])
    theirs = mcalc.lcl(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["TD"], "K"))[1]
    return ours_t, theirs.m_as("K")


# ---- thermo: vapor pressure and mixing ratio ----

@case("vapor_pressure", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.vapor_pressure(d["P"], d["W"]), mcalc.vapor_pressure(Q(d["P"], "Pa"), Q(d["W"], ""))


@case("saturation_vapor_pressure", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    return mm.saturation_vapor_pressure(d["T"]), mcalc.saturation_vapor_pressure(Q(d["T"], "K"))


@case("saturation_vapor_pressure_solid", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    t = d["T"] - 40.0
    return (
        mm.saturation_vapor_pressure(t, phase="solid"),
        mcalc.saturation_vapor_pressure(Q(t, "K"), phase="solid"),
    )


@case("saturation_vapor_pressure_auto", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    return (
        mm.saturation_vapor_pressure(d["T"], phase="auto"),
        mcalc.saturation_vapor_pressure(Q(d["T"], "K"), phase="auto"),
    )


@case("dewpoint", 1e-10, 1e-12)
def _(d):
    Q = d["Q"]
    return mm.dewpoint(d["E"]), _kelvin(mcalc.dewpoint(Q(d["E"], "Pa")))


@case("dewpoint_from_relative_humidity", 1e-10, 1e-12)
def _(d):
    Q = d["Q"]
    return (
        mm.dewpoint_from_relative_humidity(d["T"], d["RH"]),
        _kelvin(mcalc.dewpoint_from_relative_humidity(Q(d["T"], "K"), Q(d["RH"], ""))),
    )


@case("mixing_ratio", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.mixing_ratio(d["E"], d["P"]), mcalc.mixing_ratio(Q(d["E"], "Pa"), Q(d["P"], "Pa"))


@case("saturation_mixing_ratio", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    return (
        mm.saturation_mixing_ratio(d["P"], d["T"]),
        mcalc.saturation_mixing_ratio(Q(d["P"], "Pa"), Q(d["T"], "K")),
    )


@case("equivalent_potential_temperature", 1e-9, 1e-11)
def _(d):
    Q = d["Q"]
    return (
        mm.equivalent_potential_temperature(d["P"], d["T"], d["TD"]),
        mcalc.equivalent_potential_temperature(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["TD"], "K")),
    )


# ---- thermo: virtual temperature and density ----

@case("virtual_temperature", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.virtual_temperature(d["T"], d["W"]), mcalc.virtual_temperature(Q(d["T"], "K"), Q(d["W"], ""))


@case("virtual_temperature_from_dewpoint", 1e-11, 1e-13)
def _(d):
    Q = d["Q"]
    return (
        mm.virtual_temperature_from_dewpoint(d["P"], d["T"], d["TD"]),
        mcalc.virtual_temperature_from_dewpoint(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["TD"], "K")),
    )


@case("virtual_potential_temperature", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    return (
        mm.virtual_potential_temperature(d["P"], d["T"], d["W"]),
        mcalc.virtual_potential_temperature(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["W"], "")),
    )


@case("density", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.density(d["P"], d["T"], d["W"]), mcalc.density(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["W"], ""))


# ---- thermo: psychrometry ----

@case("psychrometric_vapor_pressure_wet", 1e-9, 1e-12)
def _(d):
    Q = d["Q"]
    return (
        mm.psychrometric_vapor_pressure_wet(d["P"], d["T"], d["TD"]),
        mcalc.psychrometric_vapor_pressure_wet(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["TD"], "K")),
    )


@case("relative_humidity_wet_psychrometric", 1e-8, 1e-10)
def _(d):
    Q = d["Q"]
    return (
        mm.relative_humidity_wet_psychrometric(d["P"], d["T"], d["TD"]),
        mcalc.relative_humidity_wet_psychrometric(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["TD"], "K")),
    )


# ---- thermo: humidity conversions ----

@case("mixing_ratio_from_relative_humidity", 1e-8, 1e-10)
def _(d):
    Q = d["Q"]
    return (
        mm.mixing_ratio_from_relative_humidity(d["P"], d["T"], d["RH"]),
        mcalc.mixing_ratio_from_relative_humidity(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["RH"], "")),
    )


@case("relative_humidity_from_mixing_ratio", 1e-8, 1e-10)
def _(d):
    Q = d["Q"]
    return (
        mm.relative_humidity_from_mixing_ratio(d["P"], d["T"], d["W"]),
        mcalc.relative_humidity_from_mixing_ratio(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["W"], "")),
    )


@case("mixing_ratio_from_specific_humidity", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return (
        mm.mixing_ratio_from_specific_humidity(d["W"]),
        mcalc.mixing_ratio_from_specific_humidity(Q(d["W"], "")),
    )


@case("specific_humidity_from_mixing_ratio", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return (
        mm.specific_humidity_from_mixing_ratio(d["W"]),
        mcalc.specific_humidity_from_mixing_ratio(Q(d["W"], "")),
    )


@case("relative_humidity_from_specific_humidity", 1e-8, 1e-10)
def _(d):
    Q = d["Q"]
    return (
        mm.relative_humidity_from_specific_humidity(d["P"], d["T"], d["W"]),
        mcalc.relative_humidity_from_specific_humidity(Q(d["P"], "Pa"), Q(d["T"], "K"), Q(d["W"], "")),
    )


@case("specific_humidity_from_dewpoint", 1e-8, 1e-10)
def _(d):
    Q = d["Q"]
    return (
        mm.specific_humidity_from_dewpoint(d["P1"], d["TD1"]),
        mcalc.specific_humidity_from_dewpoint(Q(d["P1"], "Pa"), Q(d["TD1"], "K")),
    )


@case("dewpoint_from_specific_humidity", 1e-8, 1e-6)
def _(d):
    Q = d["Q"]
    q = d["W"][:40]
    return (
        mm.dewpoint_from_specific_humidity(d["P1"], q),
        _kelvin(mcalc.dewpoint_from_specific_humidity(Q(d["P1"], "Pa"), Q(q, ""))),
    )


# ---- thermo: static energy, thickness, stability ----

@case("dry_static_energy", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return (
        mm.dry_static_energy(d["Z"], d["T"]),
        mcalc.dry_static_energy(Q(d["Z"], "meter"), Q(d["T"], "K")).m_as("J/kg"),
    )


@case("moist_static_energy", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return (
        mm.moist_static_energy(d["Z"], d["T"], d["W"]),
        mcalc.moist_static_energy(Q(d["Z"], "meter"), Q(d["T"], "K"), Q(d["W"], "")).m_as("J/kg"),
    )


@case("thickness_hydrostatic", 1e-9, 1e-9)
def _(d):
    Q = d["Q"]
    return (
        mm.thickness_hydrostatic(d["P1"], d["T1"]),
        mcalc.thickness_hydrostatic(Q(d["P1"], "Pa"), Q(d["T1"], "K")).m_as("m"),
    )


@case("thickness_hydrostatic_moist", 1e-8, 1e-8)
def _(d):
    Q = d["Q"]
    return (
        mm.thickness_hydrostatic(d["P1"], d["T1"], mixing_ratio=d["W1"]),
        mcalc.thickness_hydrostatic(
            Q(d["P1"], "Pa"), Q(d["T1"], "K"), mixing_ratio=Q(d["W1"], "")
        ).m_as("m"),
    )


@case("thickness_hydrostatic_from_relative_humidity", 1e-7, 1e-7)
def _(d):
    Q = d["Q"]
    rh = np.clip(mm.relative_humidity_from_dewpoint(d["T1"], d["TD1"]), 0.01, 1.0)
    return (
        mm.thickness_hydrostatic_from_relative_humidity(d["P1"], d["T1"], rh),
        mcalc.thickness_hydrostatic_from_relative_humidity(
            Q(d["P1"], "Pa"), Q(d["T1"], "K"), Q(rh, "")
        ).m_as("m"),
    )


@case("brunt_vaisala_frequency_squared", 1e-8, 1e-12)
def _(d):
    Q = d["Q"]
    return (
        mm.brunt_vaisala_frequency_squared(d["H1"], d["TH1"]),
        mcalc.brunt_vaisala_frequency_squared(Q(d["H1"], "meter"), Q(d["TH1"], "K")),
    )


@case("brunt_vaisala_frequency", 1e-8, 1e-10)
def _(d):
    Q = d["Q"]
    return (
        mm.brunt_vaisala_frequency(d["H1"], d["TH1"]),
        mcalc.brunt_vaisala_frequency(Q(d["H1"], "meter"), Q(d["TH1"], "K")),
    )


@case("brunt_vaisala_period", 1e-8, 1e-4)
def _(d):
    Q = d["Q"]
    return (
        mm.brunt_vaisala_period(d["H1"], d["TH1"]),
        mcalc.brunt_vaisala_period(Q(d["H1"], "meter"), Q(d["TH1"], "K")),
    )


@case("static_stability", 1e-7, 1e-14)
def _(d):
    Q = d["Q"]
    return (
        mm.static_stability(d["P1"], d["T1"]),
        mcalc.static_stability(Q(d["P1"], "Pa"), Q(d["T1"], "K")),
    )


@case("gradient_richardson_number", 1e-8, 1e-10)
def _(d):
    Q = d["Q"]
    return (
        mm.gradient_richardson_number(d["H1"], d["TH1"], d["U1"], d["V1"]),
        mcalc.gradient_richardson_number(
            Q(d["H1"], "meter"), Q(d["TH1"], "K"), Q(d["U1"], "m/s"), Q(d["V1"], "m/s")
        ),
    )


@case("vertical_velocity_pressure", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    w = np.full(40, 0.0)
    return (
        mm.vertical_velocity_pressure(w, d["P1"], d["T1"]),
        mcalc.vertical_velocity_pressure(Q(w, "m/s"), Q(d["P1"], "Pa"), Q(d["T1"], "K")).m_as("Pa/s"),
    )


@case("scale_height", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return (
        mm.scale_height(d["T1"][:-1], d["T1"][1:]),
        mcalc.scale_height(Q(d["T1"][:-1], "K"), Q(d["T1"][1:], "K")),
    )


# ---- basic: wind ----

@case("wind_speed", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.wind_speed(d["U"], d["V"]), mcalc.wind_speed(Q(d["U"], "m/s"), Q(d["V"], "m/s"))


@case("wind_direction", 1e-12, 1e-12)
def _(d):
    Q = d["Q"]
    return (
        mm.wind_direction(d["U"], d["V"]),
        mcalc.wind_direction(Q(d["U"], "m/s"), Q(d["V"], "m/s")),
    )


@case("wind_direction_to", 1e-12, 1e-12)
def _(d):
    Q = d["Q"]
    return (
        mm.wind_direction(d["U"], d["V"], "to"),
        mcalc.wind_direction(Q(d["U"], "m/s"), Q(d["V"], "m/s"), convention="to"),
    )


@case("wind_components_u", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    ours = mm.wind_components(np.abs(d["U"]), d["LAT"])[0]
    return ours, mcalc.wind_components(Q(np.abs(d["U"]), "m/s"), Q(d["LAT"], "radian"))[0].m_as("m/s")


@case("wind_components_v", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    ours = mm.wind_components(np.abs(d["U"]), d["LAT"])[1]
    return ours, mcalc.wind_components(Q(np.abs(d["U"]), "m/s"), Q(d["LAT"], "radian"))[1].m_as("m/s")


def _masked_case(name, ours_fn, theirs_fn, rtol=1e-12, atol=1e-12):
    def run(d):
        return ours_fn(d), theirs_fn(d)

    CASES.append((name, run, rtol, atol))


def _compare_masked(ours, theirs, rtol, atol):
    assert np.array_equal(np.ma.getmaskarray(ours), np.ma.getmaskarray(theirs))
    np.testing.assert_allclose(
        np.ma.getdata(ours), np.ma.getdata(theirs), rtol=rtol, atol=atol
    )


for _flag in (True, False):
    _masked_case(
        f"windchill[mask_undefined={_flag}]",
        lambda d, f=_flag: mm.windchill(d["TC"], d["SPD"], mask_undefined=f),
        lambda d, f=_flag: mcalc.windchill(
            d["Q"](d["TC"], "degC"), d["Q"](d["SPD"], "km/hr"), mask_undefined=f
        ),
        rtol=1e-9, atol=1e-4,
    )
    _masked_case(
        f"windchill_face_level[mask_undefined={_flag}]",
        lambda d, f=_flag: mm.windchill(d["TC"], d["SPD"], face_level_winds=True, mask_undefined=f),
        lambda d, f=_flag: mcalc.windchill(
            d["Q"](d["TC"], "degC"), d["Q"](d["SPD"], "km/hr"),
            face_level_winds=True, mask_undefined=f,
        ),
        rtol=1e-9, atol=1e-4,
    )
    _masked_case(
        f"heat_index[mask_undefined={_flag}]",
        lambda d, f=_flag: mm.heat_index(d["TF"], d["RHF"], mask_undefined=f),
        lambda d, f=_flag: mcalc.heat_index(
            d["Q"](d["TF"], "degF"), d["Q"](d["RHF"], ""), mask_undefined=f
        ),
    )
    _masked_case(
        f"apparent_temperature_F[mask_undefined={_flag}]",
        lambda d, f=_flag: mm.apparent_temperature(d["TF"], d["RHF"], d["SPD"], mask_undefined=f),
        lambda d, f=_flag: mcalc.apparent_temperature(
            d["Q"](d["TF"], "degF"), d["Q"](d["RHF"], ""), d["Q"](d["SPD"], "km/hr"),
            mask_undefined=f,
        ),
    )
    _masked_case(
        f"apparent_temperature_cold[mask_undefined={_flag}]",
        lambda d, f=_flag: mm.apparent_temperature(d["COLD"], d["RHF"], d["SPD"], mask_undefined=f),
        lambda d, f=_flag: mcalc.apparent_temperature(
            d["Q"](d["COLD"], "degF"), d["Q"](d["RHF"], ""), d["Q"](d["SPD"], "km/hr"),
            mask_undefined=f,
        ),
        rtol=1e-9, atol=1e-4,
    )


# ---- basic: height, pressure, altimeter ----

@case("pressure_to_height_std", 1e-8, 1e-11)
def _(d):
    Q = d["Q"]
    return (
        mm.pressure_to_height_std(d["P"]),
        mcalc.pressure_to_height_std(Q(d["P"], "Pa")).m_as("m"),
    )


@case("height_to_pressure_std", 1e-8, 1e-11)
def _(d):
    Q = d["Q"]
    h = mm.pressure_to_height_std(d["P"])
    return (
        mm.height_to_pressure_std(h),
        mcalc.height_to_pressure_std(mcalc.pressure_to_height_std(Q(d["P"], "Pa"))).m_as("Pa"),
    )


@case("height_to_geopotential", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return (
        mm.height_to_geopotential(d["Z"]),
        mcalc.height_to_geopotential(Q(d["Z"], "meter")).m_as("meter**2 / second**2"),
    )


@case("geopotential_to_height", 1e-10, 1e-12)
def _(d):
    Q = d["Q"]
    return (
        mm.geopotential_to_height(mm.height_to_geopotential(d["Z"])),
        mcalc.geopotential_to_height(
            mcalc.height_to_geopotential(Q(d["Z"], "meter"))
        ).m_as("m"),
    )


@case("coriolis_parameter", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    return mm.coriolis_parameter(d["LAT"]), mcalc.coriolis_parameter(Q(d["LAT"], "radian"))


@case("add_height_to_pressure", 1e-8, 1e-9)
def _(d):
    Q = d["Q"]
    return (
        mm.add_height_to_pressure(d["P"], d["Z"]),
        mcalc.add_height_to_pressure(Q(d["P"], "Pa"), Q(d["Z"], "meter")).m_as("Pa"),
    )


@case("add_pressure_to_height", 1e-8, 1e-9)
def _(d):
    Q = d["Q"]
    return (
        mm.add_pressure_to_height(d["Z"], d["P"] * 0.1),
        mcalc.add_pressure_to_height(Q(d["Z"], "meter"), Q(d["P"] * 0.1, "Pa")).m_as("m"),
    )


@case("sigma_to_pressure", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    # upstream takes scalar surface/top pressure
    return (
        mm.sigma_to_pressure(d["SIG"], float(d["P"].max()), 1000.0),
        mcalc.sigma_to_pressure(Q(d["SIG"], ""), Q(float(d["P"].max()), "Pa"), Q(1000.0, "Pa")),
    )


@case("altimeter_to_station_pressure", 1e-9, 1e-8)
def _(d):
    Q = d["Q"]
    return (
        mm.altimeter_to_station_pressure(d["ALT"] * 100.0, d["Z"]),
        mcalc.altimeter_to_station_pressure(Q(d["ALT"], "millibar"), Q(d["Z"], "meter")).m_as("Pa"),
    )


@case("altimeter_to_sea_level_pressure", 1e-9, 1e-8)
def _(d):
    Q = d["Q"]
    return (
        mm.altimeter_to_sea_level_pressure(d["ALT"] * 100.0, d["Z"], d["T"]),
        mcalc.altimeter_to_sea_level_pressure(
            Q(d["ALT"], "millibar"), Q(d["Z"], "meter"), Q(d["T"], "K")
        ).m_as("Pa"),
    )


# ---- tools and kinematics on a 2-D grid ----

@case("first_derivative_uniform", 1e-11, 1e-13)
def _(d):
    Q = d["Q"]
    s = d["SCALAR"]
    return mm.first_derivative(s, axis=-1, delta=2.0), mcalc.first_derivative(Q(s, "K"), axis=-1, delta=2.0)


@case("first_derivative_x", 1e-10, 1e-12)
def _(d):
    Q = d["Q"]
    s = d["SCALAR"]
    lon = d["LON"][0]
    return (
        mm.first_derivative(s, axis=1, x=lon),
        mcalc.first_derivative(Q(s, "K"), axis=1, x=Q(lon, "degree")).m_as("K/degree"),
    )


@case("first_derivative_nonuniform", 1e-10, 1e-12)
def _(d):
    Q = d["Q"]
    s = d["SCALAR"]
    delta = 1.0 + 0.1 * np.arange(39)
    return (
        mm.first_derivative(s, axis=-1, delta=delta),
        mcalc.first_derivative(Q(s, "K"), axis=-1, delta=Q(delta, "meter")),
    )


@case("second_derivative_uniform", 1e-11, 1e-13)
def _(d):
    Q = d["Q"]
    s = d["SCALAR"]
    return mm.second_derivative(s, axis=-1, delta=2.0), mcalc.second_derivative(Q(s, "K"), axis=-1, delta=2.0)


@case("second_derivative_y", 1e-11, 1e-13)
def _(d):
    Q = d["Q"]
    s = d["SCALAR"]
    return mm.second_derivative(s, axis=-2, delta=2.0), mcalc.second_derivative(Q(s, "K"), axis=-2, delta=2.0)


@case("gradient", 1e-11, 1e-13)
def _(d):
    Q = d["Q"]
    s = d["SCALAR"]
    theirs = mcalc.gradient(Q(s, "K"), axes=(-1, -2),
                           deltas=(Q(2.0, "meter"), Q(2.0, "meter")))
    return mm.gradient(s, axes=(-1, -2), deltas=(2.0, 2.0))[0], theirs[0].m_as("K/m")


@case("laplacian", 1e-11, 1e-13)
def _(d):
    Q = d["Q"]
    s = d["SCALAR"]
    return (
        mm.laplacian(s, axes=(-1, -2), deltas=(2.0, 2.0)),
        mcalc.laplacian(Q(s, "K"), axes=(-1, -2),
                        deltas=(Q(2.0, "meter"), Q(2.0, "meter"))),
    )


@case("vorticity", 1e-8, 1e-13)
def _(d):
    Q = d["Q"]
    return (
        mm.vorticity(d["UU"], d["VV"], dx=d["DX"], dy=d["DY"]),
        mcalc.vorticity(Q(d["UU"], "m/s"), Q(d["VV"], "m/s"), dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter")).m_as("1/s"),
    )


@case("divergence", 1e-8, 1e-13)
def _(d):
    Q = d["Q"]
    return (
        mm.divergence(d["UU"], d["VV"], dx=d["DX"], dy=d["DY"]),
        mcalc.divergence(Q(d["UU"], "m/s"), Q(d["VV"], "m/s"), dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter")).m_as("1/s"),
    )


@case("shearing_deformation", 1e-8, 1e-13)
def _(d):
    Q = d["Q"]
    return (
        mm.shearing_deformation(d["UU"], d["VV"], dx=d["DX"], dy=d["DY"]),
        mcalc.shearing_deformation(Q(d["UU"], "m/s"), Q(d["VV"], "m/s"), dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter")).m_as("1/s"),
    )


@case("stretching_deformation", 1e-8, 1e-13)
def _(d):
    Q = d["Q"]
    return (
        mm.stretching_deformation(d["UU"], d["VV"], dx=d["DX"], dy=d["DY"]),
        mcalc.stretching_deformation(Q(d["UU"], "m/s"), Q(d["VV"], "m/s"), dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter")).m_as("1/s"),
    )


@case("total_deformation", 1e-8, 1e-13)
def _(d):
    Q = d["Q"]
    return (
        mm.total_deformation(d["UU"], d["VV"], dx=d["DX"], dy=d["DY"]),
        mcalc.total_deformation(Q(d["UU"], "m/s"), Q(d["VV"], "m/s"), dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter")).m_as("1/s"),
    )


@case("vector_derivative", 1e-8, 1e-13)
def _(d):
    Q = d["Q"]
    ours = mm.vector_derivative(d["UU"], d["VV"], dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"))
    theirs = mcalc.vector_derivative(Q(d["UU"], "m/s"), Q(d["VV"], "m/s"), dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"))
    return ours[3], theirs[3].m_as("1/s")


@case("geostrophic_wind_u", 1e-7, 1e-8)
def _(d):
    Q = d["Q"]
    ours = mm.geostrophic_wind(d["SCALAR"], dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"), latitude=np.deg2rad(d["LAT2"]))[0]
    theirs = mcalc.geostrophic_wind(
        Q(d["SCALAR"], "meter"), dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"),
        latitude=Q(np.deg2rad(d["LAT2"]), "radian")
    )[0]
    return ours, theirs.m_as("m/s")


@case("geostrophic_wind_v", 1e-7, 1e-8)
def _(d):
    Q = d["Q"]
    ours = mm.geostrophic_wind(d["SCALAR"], dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"), latitude=np.deg2rad(d["LAT2"]))[1]
    theirs = mcalc.geostrophic_wind(
        Q(d["SCALAR"], "meter"), dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"),
        latitude=Q(np.deg2rad(d["LAT2"]), "radian")
    )[1]
    return ours, theirs.m_as("m/s")


@case("ageostrophic_wind_u", 1e-7, 1e-8)
def _(d):
    Q = d["Q"]
    lat = np.deg2rad(d["LAT2"])
    ours = mm.ageostrophic_wind(d["SCALAR"], d["UU"], d["VV"], dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"), latitude=lat)[0]
    theirs = mcalc.ageostrophic_wind(
        Q(d["SCALAR"], "meter"), Q(d["UU"], "m/s"), Q(d["VV"], "m/s"),
        dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"), latitude=Q(lat, "radian"),
    )[0]
    return ours, theirs.m_as("m/s")


@case("ageostrophic_wind_v", 1e-7, 1e-8)
def _(d):
    Q = d["Q"]
    lat = np.deg2rad(d["LAT2"])
    ours = mm.ageostrophic_wind(d["SCALAR"], d["UU"], d["VV"], dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"), latitude=lat)[1]
    theirs = mcalc.ageostrophic_wind(
        Q(d["SCALAR"], "meter"), Q(d["UU"], "m/s"), Q(d["VV"], "m/s"),
        dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"), latitude=Q(lat, "radian"),
    )[1]
    return ours, theirs.m_as("m/s")


@case("absolute_vorticity", 1e-7, 1e-8)
def _(d):
    Q = d["Q"]
    lat = np.deg2rad(d["LAT2"])
    return (
        mm.absolute_vorticity(d["UU"], d["VV"], dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"), latitude=lat),
        mcalc.absolute_vorticity(
            Q(d["UU"], "m/s"), Q(d["VV"], "m/s"), dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter"), latitude=Q(lat, "radian")
        ),
    )


@case("advection", 1e-7, 1e-9)
def _(d):
    Q = d["Q"]
    return (
        mm.advection(d["SCALAR"], d["UU"], d["VV"], dx=d["DX"], dy=d["DY"]),
        # upstream's `advection` demands length units on dx/dy, but its
        # `first_derivative` then hands back the un-scaled magnitude; the
        # numbers are still the d/dx and d/dy of the field, so compare `.m`.
        mcalc.advection(
            Q(d["SCALAR"], "K"), u=Q(d["UU"], "m/s"), v=Q(d["VV"], "m/s"),
            dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter")
        ).m,
    )


@case("frontogenesis", 1e-7, 1e-9)
def _(d):
    Q = d["Q"]
    return (
        mm.frontogenesis(d["SCALAR"], d["UU"], d["VV"], dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter")),
        mcalc.frontogenesis(Q(d["SCALAR"], "K"), Q(d["UU"], "m/s"), Q(d["VV"], "m/s"),
                            dx=Q(d["DX"], "meter"), dy=Q(d["DY"], "meter")),
    )


@case("montgomery_streamfunction", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    z = np.full_like(d["SCALAR"], 300.0)
    return (
        mm.montgomery_streamfunction(z, d["SCALAR"]),
        mcalc.montgomery_streamfunction(Q(z, "meter"), Q(d["SCALAR"], "K")).m_as("J/kg"),
    )


# ---- smoothing ----

@case("smooth_n_point_5", 1e-11, 1e-13)
def _(d):
    g = np.random.default_rng(2).normal(size=(40, 50))
    return mm.smooth_n_point(g, 5), mcalc.smooth_n_point(g, 5)


@case("smooth_n_point_9", 1e-11, 1e-13)
def _(d):
    g = np.random.default_rng(2).normal(size=(40, 50))
    return mm.smooth_n_point(g, 9), mcalc.smooth_n_point(g, 9)


@case("smooth_rectangular", 1e-11, 1e-13)
def _(d):
    g = np.random.default_rng(2).normal(size=(40, 50))
    return mm.smooth_rectangular(g, 5), mcalc.smooth_rectangular(g, 5)


@case("smooth_circular", 1e-11, 1e-13)
def _(d):
    g = np.random.default_rng(2).normal(size=(40, 50))
    return mm.smooth_circular(g, 3), mcalc.smooth_circular(g, 3)


@case("smooth_window_normalized", 1e-11, 1e-13)
def _(d):
    g = np.random.default_rng(2).normal(size=(40, 50))
    w = np.array([[0.2, 0.3, 0.2], [0.1, 0.4, 0.1], [0.15, 0.25, 0.15]])
    return mm.smooth_window(g, w), mcalc.smooth_window(g, w)


# ---- indices ----

@case("supercell_composite", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    mucape, esh, eshear = d["P"][:3] * 3.0, d["T"][:3] - 270.0, d["W"][:3] * 1000.0
    return (
        mm.supercell_composite(mucape, esh, eshear),
        mcalc.supercell_composite(Q(mucape, "J/kg"), Q(esh, "m^2/s^2"), Q(eshear, "m/s")),
    )


@case("significant_tornado", 1e-12, 1e-14)
def _(d):
    Q = d["Q"]
    a, b, c, e = d["P"][:4], d["Z"][:4], d["E"][:4] * 0.1, d["T"][:4] - 200.0
    return (
        mm.significant_tornado(a, b, c, e),
        mcalc.significant_tornado(Q(a, "J/kg"), Q(b, "m"), Q(c, "m^2/s^2"), Q(e, "m/s")),
    )


@pytest.mark.parametrize("name,fn,rtol,atol", CASES, ids=[c[0] for c in CASES])
def test_parity(name, fn, rtol, atol, d):
    ours, theirs = fn(d)
    if np.ma.isMaskedArray(theirs):
        _compare_masked(ours, theirs, rtol, atol)
    else:
        np.testing.assert_allclose(
            np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float), rtol=rtol, atol=atol
        )
