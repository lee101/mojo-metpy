"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")

# `MOJOMETPY_LIB` points at an already-built `capi.so`. Two cases need it: an
# ordinary (non-editable) install, where the package is copied into
# site-packages and `ROOT` no longer names the checkout that holds `src/` and
# `build/`; and any server that must never pay a `mojo build` inside a request
# and would rather ship the kernel next to the deploy.
LIB = os.environ.get("MOJOMETPY_LIB") or os.path.join(ROOT, "build", "capi.so")

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype).  The argument list matches the Mojo `def` in
# `src/capi.mojo` exactly; keeping the two in step is the whole FFI contract.
# Regenerate with `python -m mojometpy._lib --rewrite-signatures`.
#
# `I` is `c_int64`, never ctypes' default `c_int`: Mojo's `Int` is 64-bit and
# the C ABI passes it in a full register, so a 32-bit argtype silently
# truncates every buffer address above 4 GiB into a wild pointer.
_SIGNATURES = {
    "mm_add_height_to_pressure": ([I] * 4, None),
    "mm_add_pressure_to_height": ([I] * 4, None),
    "mm_advection": ([I] * 9, None),
    "mm_altimeter_to_sea_level_pressure": ([I] * 5, None),
    "mm_altimeter_to_station_pressure": ([I] * 4, None),
    "mm_apparent_temperature": ([I] * 8, None),
    "mm_brunt_vaisala_frequency": ([I] * 3, None),
    "mm_brunt_vaisala_frequency_squared": ([I] * 5, None),
    "mm_brunt_vaisala_period": ([I] * 3, None),
    "mm_coriolis_parameter": ([I] * 3, None),
    "mm_curvature_vorticity": ([I] * 8, None),
    "mm_density": ([I, I, I, F, I, I], None),
    "mm_dewpoint": ([I] * 3, None),
    "mm_dewpoint_from_relative_humidity": ([I] * 4, None),
    "mm_dewpoint_from_specific_humidity": ([I] * 4, None),
    "mm_divergence": ([I] * 4, None),
    "mm_dry_lapse": ([I] * 5, None),
    "mm_dry_static_energy": ([I] * 4, None),
    "mm_equivalent_potential_temperature": ([I] * 5, None),
    "mm_exner_function": ([I, F, I, I], None),
    "mm_first_derivative": ([I] * 7, None),
    "mm_friction_velocity": ([I] * 9, None),
    "mm_frontogenesis": ([I] * 8, None),
    "mm_geopotential_to_height": ([I] * 3, None),
    "mm_geostrophic_wind": ([I] * 6, None),
    "mm_gradient_richardson_number": ([I] * 6, None),
    "mm_heat_index": ([I] * 6, None),
    "mm_height_to_geopotential": ([I] * 3, None),
    "mm_height_to_pressure_std": ([I] * 3, None),
    "mm_inertial_advective_wind": ([I] * 10, None),
    "mm_interpolate_1d": ([I, I, I, I, I, I, I, F, I], I),
    "mm_lat_lon_grid_deltas_dx": ([I] * 6, None),
    "mm_lat_lon_grid_deltas_dy": ([I] * 6, None),
    "mm_lcl": ([I] * 6, None),
    "mm_mixing_ratio": ([I, I, F, I, I], None),
    "mm_mixing_ratio_from_relative_humidity": ([I] * 6, None),
    "mm_mixing_ratio_from_specific_humidity": ([I] * 3, None),
    "mm_moist_air_gas_constant": ([I] * 3, None),
    "mm_moist_air_poisson_exponent": ([I] * 3, None),
    "mm_moist_air_specific_heat_pressure": ([I] * 3, None),
    "mm_moist_static_energy": ([I] * 5, None),
    "mm_montgomery_streamfunction": ([I] * 4, None),
    "mm_nearest_intersection_idx": ([I] * 4, I),
    "mm_normalize_window": ([I] * 2, I),
    "mm_potential_temperature": ([I] * 4, None),
    "mm_potential_vorticity_baroclinic": ([I] * 8, None),
    "mm_potential_vorticity_barotropic": ([I] * 4, None),
    "mm_pressure_to_height_std": ([I] * 3, None),
    "mm_psychrometric_vapor_pressure_wet": ([I, I, I, F, I, I], None),
    "mm_q_vector": ([I, I, I, I, I, I, I, F, I, I, I], None),
    "mm_relative_humidity_from_dewpoint": ([I] * 5, None),
    "mm_relative_humidity_from_mixing_ratio": ([I] * 6, None),
    "mm_relative_humidity_from_specific_humidity": ([I] * 6, None),
    "mm_relative_humidity_wet_psychrometric": ([I, I, I, F, I, I], None),
    "mm_resample_nn_1d_nearest": ([I, F, I], I),
    "mm_saturation_mixing_ratio": ([I] * 5, None),
    "mm_saturation_vapor_pressure": ([I] * 4, None),
    "mm_saturation_vapor_pressure_liquid": ([I] * 3, None),
    "mm_saturation_vapor_pressure_solid": ([I] * 3, None),
    "mm_scale_height": ([I] * 4, None),
    "mm_second_derivative": ([I] * 7, None),
    "mm_shear_vorticity": ([I] * 8, None),
    "mm_shearing_deformation": ([I] * 4, None),
    "mm_sigma_to_pressure": ([I] * 5, None),
    "mm_significant_tornado": ([I] * 6, None),
    "mm_smooth_window": ([I] * 8, None),
    "mm_specific_humidity_from_dewpoint": ([I] * 5, None),
    "mm_specific_humidity_from_mixing_ratio": ([I] * 3, None),
    "mm_static_stability": ([I] * 5, None),
    "mm_storm_relative_helicity": ([I, I, F, F, I, I], None),
    "mm_stretching_deformation": ([I] * 4, None),
    "mm_supercell_composite": ([I] * 5, None),
    "mm_temperature_from_potential_temperature": ([I] * 4, None),
    "mm_thickness_hydrostatic": ([I, I, I, I, F, I, I], None),
    "mm_thickness_hydrostatic_from_relative_humidity": ([I] * 5, None),
    "mm_tke": ([I] * 8, None),
    "mm_total_deformation": ([I] * 6, None),
    "mm_vapor_pressure": ([I] * 4, None),
    "mm_vector_derivative": ([I] * 14, None),
    "mm_vector_derivative_map_correction": ([I] * 11, None),
    "mm_vertical_velocity_pressure": ([I] * 6, None),
    "mm_virtual_potential_temperature": ([I, I, I, F, I, I], None),
    "mm_virtual_temperature": ([I, I, F, I, I], None),
    "mm_virtual_temperature_from_dewpoint": ([I, I, I, F, I, I, I], None),
    "mm_vorticity": ([I] * 4, None),
    "mm_water_latent_heat_melting": ([I] * 3, None),
    "mm_water_latent_heat_sublimation": ([I] * 3, None),
    "mm_water_latent_heat_vaporization": ([I] * 3, None),
    "mm_wind_components": ([I] * 5, None),
    "mm_wind_direction": ([I] * 5, None),
    "mm_wind_speed": ([I] * 4, None),
    "mm_windchill": ([I] * 7, None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOMETPY_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOMETPY_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `build/capi.so` if it is missing or stale."""
    if os.environ.get("MOJOMETPY_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not os.path.isdir(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo sources at {SRC} and no library at {LIB}; this looks like "
            "a copied install - point MOJOMETPY_LIB at a built capi.so"
        )
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    env = dict(os.environ)
    if not env.get("MODULAR_HOME"):
        env["MODULAR_HOME"] = modular_home()
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def capi_source() -> str:
    """The Mojo source that defines the C ABI, as text."""
    return read_capi_source()


def read_capi_source() -> str:
    """`src/capi.mojo` as text, read once."""
    global _capi_source
    if _capi_source is None:
        with open(os.path.join(SRC, "capi.mojo")) as handle:
            _capi_source = handle.read()
    return _capi_source


def modular_home() -> str:
    """`MODULAR_HOME` must name the env's `share/max` or every import fails.

    `pixi run` sets it; a bare `mojo build` does not, so derive it from wherever
    the `mojo` binary that is about to run actually lives.
    """
    exe = os.environ.get("MOJOMETPY_MOJO", "").split()[:1]
    exe = exe[0] if exe else shutil.which("mojo")
    if exe:
        share = os.path.join(os.path.dirname(os.path.dirname(exe)), "share", "max")
        if os.path.isdir(share):
            return share
    raise BuildError("cannot locate share/max; set MODULAR_HOME or MOJOMETPY_MOJO")


_lib = None
_capi_source = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def parse_signatures(path: str | None = None) -> dict:
    """Read the `@export` list out of `src/capi.mojo`.

    The Mojo source is the single source of truth for the C ABI: this reads every
    `abi("C")` export and returns `name -> (argtypes, restype)`, so the ctypes
    table cannot drift out of step with the kernels it calls.  A `Float64`
    argument becomes `c_double`, every other argument becomes `c_int64`, and a
    `-> Int` return becomes `c_int64`.
    """
    if path is None:
        src = read_capi_source()
    else:
        with open(path) as handle:
            src = handle.read()
    exports = {}
    for match in re.finditer(r'@export\("([A-Za-z_0-9]+)"\)', src):
        exports[match.group(1)] = match.end()
    out = {}
    for sym, after in exports.items():
        tail = src[after:after + 800]
        head = re.match(r'\s*def\s+[A-Za-z_0-9]+\s*\(', tail)
        if head is None:
            raise BuildError(f"{sym} is not followed by a def")
        args, _, _ = _split_call(tail[head.end():])
        if not args:
            raise BuildError(f"{sym} takes no arguments")
        # the return type is written after the `abi("C")` effect, not before the colon
        ret = re.search(r'\)\s*abi\("C"\)\s*->\s*([A-Za-z_0-9]+)', tail)
        out[sym] = ([_ctype(t) for _, t in args], I if ret else None)
    if not out:
        raise BuildError(f'no abi("C") exports found in {path}')
    return out


def _ctype(mojo_type: str):
    """The ctypes counterpart of one Mojo parameter type."""
    if mojo_type in ("Float64", "Float32"):
        return F
    if mojo_type == "Int":
        return I
    raise BuildError(f"unmapped C ABI type {mojo_type!r}")


def _split_call(text: str) -> tuple[list[tuple[str, str]], str, str]:
    """Split `(a: Int, b: Float64) -> Int:` into `(name, type)` pairs and the return type."""
    close = text.index(")")
    args = []
    for piece in text[:close].replace("\n", " ").split(","):
        piece = piece.strip()
        if not piece:
            continue
        name, _, kind = piece.partition(":")
        args.append((name.strip(), kind.strip()))
    rest = text[close + 1:]
    match = re.match(r"\s*->\s*([A-Za-z_0-9]+)\s*:", rest)
    return args, text[:close], match.group(1) if match else ""


def rewrite_signatures(path: str | None = None) -> None:
    """Rewrite `_SIGNATURES` in this file from the exports in `src/capi.mojo`."""
    path = path or os.path.abspath(__file__)
    sig = parse_signatures()
    lines = [
        "# name -> (argtypes, restype).  The argument list matches the Mojo `def` in",
        "# `src/capi.mojo` exactly; keeping the two in step is the whole FFI contract.",
        "# Regenerate with `python -m mojometpy._lib --rewrite-signatures`.",
        "#",
        "# `I` is `c_int64`, never ctypes' default `c_int`: Mojo's `Int` is 64-bit and",
        "# the C ABI passes it in a full register, so a 32-bit argtype silently",
        "# truncates every buffer address above 4 GiB into a wild pointer.",
        "_SIGNATURES = {",
    ]
    for name in sorted(sig):
        argtypes, restype = sig[name]
        tail = "I" if restype is I else "None"
        if any(a is F for a in argtypes):
            spec = ", ".join("F" if a is F else "I" for a in argtypes)
            lines.append(f'    "{name}": ([{spec}], {tail}),')
        elif restype is I:
            lines.append(f'    "{name}": ([I] * {len(argtypes)}, I),')
        else:
            lines.append(f'    "{name}": ([I] * {len(argtypes)}, None),')
    lines.append("}")
    with open(path) as handle:
        src = handle.read()
    start = src.index("# name -> (argtypes, restype).")
    end = src.index("class BuildError")
    with open(path, "w") as handle:
        handle.write(src[:start] + "\n".join(lines) + "\n\n\n" + src[end:])


def main() -> int:
    """`python -m mojometpy._lib` rebuilds the library, or prints the ABI table."""
    if "--rewrite-signatures" in sys.argv:
        rewrite_signatures()
        return 0
    if "--signatures" in sys.argv:
        for name, (argtypes, restype) in sorted(parse_signatures().items()):
            print(name, len(argtypes), restype and restype.__name__)
        return 0
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
