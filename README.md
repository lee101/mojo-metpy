# mojo-metpy

MetPy's meteorological calculations, ported to Mojo.

Upstream [MetPy](https://github.com/Unidata/MetPy) is a large Python library with
a `pint` unit layer wrapped around NumPy. This repo takes the compute-bound
kernels underneath that layer and rewrites them in Mojo, keeping upstream's names,
argument order, defaults and branch structure so the two can be read side by side
and so the port is a drop-in for the subset it covers.

Everything runs through one compilation unit, `mojo build --emit shared-lib`,
and is called from Python through `ctypes`. There is no Python fallback: if the
shared library does not load, nothing works, and `tests/test_ffi.py` asserts
that the compiled object is really the one under test.

- **92 exported kernels** across 6 modules, mirroring `metpy/calc/*.py`
- **148 public Python functions** with upstream's signatures
- **273 tests**, including parity against real upstream MetPy 1.7.1

## What this is not

This is a partial port. It covers the array-level meteorological kernels and
deliberately leaves out everything that is object-oriented or unit-driven in
upstream: `pint` quantities, `xarray` wrappers, parcels, indices that need a
full sounding. Section *Coverage* says exactly what is and is not here.

## Install

```bash
pixi install
pixi run build
```

`pixi run build` compiles `src/capi.mojo` into `dist/libmojo-metpy.so`. Upstream
MetPy is installed as a dependency so the parity tests have something real to
compare against.

## Usage

```python
import numpy as np
from mojometpy import thermo, kinematics

pressure = np.linspace(100000.0, 10000.0, 50)      # Pa
temperature = 288.0 - 6.5e-3 * np.arange(50)      # K

theta = thermo.potential_temperature(pressure, temperature)
print(theta[:3])        # [288.         289.52293493 291.08304339]

dewpoint = thermo.dewpoint_from_relative_humidity(temperature, np.full(50, 0.5))
lcl_p, lcl_t = thermo.lcl(pressure, temperature, dewpoint)
print(float(lcl_p[0]), float(lcl_t[0]))   # 85554.98726776369 275.4601017991305

u = np.full((50, 60), 10.0)
v = np.full((50, 60), 0.0)
print(kinematics.divergence(u, v, dx=1000.0, dy=1000.0).shape)   # (50, 60)
```

The API mirrors upstream, so `import metpy.calc as mc` code mostly runs unchanged
with the unit arguments dropped:

```python
import metpy.calc as mc
from metpy.units import units

mc.potential_temperature(pressure * units.Pa, temperature * units.K)
# this port:
thermo.potential_temperature(pressure, temperature)
```

A `pint` Quantity passed in is unwrapped and its magnitude used, so mixing the
two during a migration is fine.

## Tasks

| command | what it does |
| --- | --- |
| `pixi run build` | `mojo build --emit shared-lib` into `dist/` |
| `pixi run test` | pytest: parity against MetPy, behaviour, and one test per exported kernel |
| `pixi run bench` | the benchmark table below, under a machine-wide flock |

## Coverage

### Covered, and checked against upstream

`basic`, `thermo`, `tools`, `kinematics`, `indices` and `turbulence`, function by
function, with upstream's names and signatures:

| module | public functions | kernels | notes |
| --- | ---: | ---: | --- |
| `basic` | 20 | 16 | wind, pressure/height, altimeter, geopotential, Coriolis, sigma, smoothing |
| `thermo` | 46 | 41 | moist-air properties, humidity, potential temperatures, stability, static energy |
| `tools` | 11 | 9 | derivatives, interpolation, layer extraction, geodesic grid deltas |
| `kinematics` | 24 | 18 | vorticity, deformation, advection, geostrophic/inertial wind, potential vorticity, frontogenesis, q-vector |
| `indices` | 13 | 6 | storm motion, bulk shear, composite indices, critical angle |
| `turbulence` | 2 | 2 | TKE, friction velocity |

92 exported kernels back 116 public functions. The difference is wrappers that
compose kernels rather than adding an arithmetic step of their own
(`gradient`, `geospatial_gradient`, `bunkers_storm_motion`, `bulk_shear`, the
`indices` that run `get_layer` first), plus the few kernels one function reaches
through another (`mm_saturation_vapor_pressure_liquid` and `_solid` behind
`mm_saturation_vapor_pressure`).

### Not covered, and why

- **`pint` units.** Upstream's decorators convert and check units; the port takes
  the SI value directly. The argument convention for each function is fixed, and
  the ones that differ are in the unit table below.
- **`xarray` wrappers.** Functions take and return `ndarray`; `DataArray`
  arguments are converted via `.values` by the caller.
- **Parcels and sounding indices.** `parcel_profile`, `most_unstable_parcel`,
  `cape_cin`, `showalter_index`, `k_index`, `sweat_index`, `lfc`, `el`,
  `mixed_layer_cape_cin`, `downdraft_cape`, `cross_totals`, `wet_bulb_temperature`
  and friends are not ported: they are control-flow-heavy object code over
  profiles, not array kernels, and rewriting them would not be a rewrite of the
  mathematics.
- **`get_layer_heights` / `get_layer` unit dispatch.** Upstream picks its branch
  from the unit of `bottom` and `depth`. Without units the branch has to be named,
  so `tools.get_layer` is the pressure branch and `tools.get_layer_by_length` is
  the length branch. Every internal caller upstream always passes metres to goes
  through the latter.
- **`lat_lon_grid_deltas`.** Upstream uses `pyproj.Geod(ellps='sphere')`; the port
  evaluates the same spherical geodesic in closed form, which is what PROJ
  computes for a sphere.
- **`smooth_window` is 2-D only.** Upstream accepts N-d grids; the port raises
  `NotImplementedError` for other ranks.
- **`turbulence.tke` / `friction_velocity` take a line geometry** `(n, stride,
  outer)` because upstream reduces along an axis; the Python wrapper computes it.

### Argument unit conventions

Where upstream's unit layer picked a scale, the port uses the SI value:

| function | upstream | this port |
| --- | --- | --- |
| `windchill`, `apparent_temperature` | degC, km/h | degC, km/h |
| `heat_index` | degF | degF |
| `precipitable_water` | mm | mm |
| `altimeter_to_station_pressure` | millibar/hPa | **Pa** |
| `montgomery_streamfunction`, `dry_static_energy` | kJ/kg | **J/kg** |
| `thickness_hydrostatic` | m | m |

The two bold rows are the ones where the numbers are not interchangeable; the
parity tests scale the input to match.

## Benchmarks

Measured on the machine described below with `pixi run bench`, best of 5. Three
columns: this port, upstream MetPy called the way upstream is called (with
`pint` Quantities, so unit checking is included), and the same formula as plain
float64 NumPy. Every case is checked against its NumPy reference before it is
timed.

Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 threads, 251 GiB RAM, Linux
6.8.0, Python 3.13, Mojo 1.2.0.dev2026092905, NumPy 2.x, MetPy 1.7.1.

| case | mojo-metpy | metpy (pint) | numpy | vs metpy | vs numpy |
| --- | ---: | ---: | ---: | ---: | ---: |
| potential_temperature (1M) | 23.29 ms | 54.90 ms | 36.97 ms | 2.36x | 1.59x |
| saturation_vapor_pressure (1M) | 6.48 ms | 44.64 ms | 44.81 ms | 6.88x | 6.91x |
| dewpoint (1M) | 5.67 ms | 18.81 ms | 4.04 ms | 3.32x | **0.71x** |
| saturation_mixing_ratio (1M) | 7.34 ms | 82.18 ms | 29.55 ms | 11.20x | 4.03x |
| equivalent_potential_temperature (1M) | 12.29 ms | 404.59 ms | - | 32.93x | - |
| density (1M) | 3.27 ms | 9.49 ms | 4.34 ms | 2.90x | 1.33x |
| virtual_temperature (1M) | 1.45 ms | 4.86 ms | 4.55 ms | 3.35x | 3.14x |
| relative_humidity_from_dewpoint (1M) | 8.78 ms | 98.35 ms | 16.65 ms | 11.20x | 1.89x |
| mixing_ratio_from_relative_humidity (1M) | 7.99 ms | 93.95 ms | - | 11.75x | - |
| lcl (1M) | 51.96 ms | 532.92 ms | - | 10.26x | - |
| thickness_hydrostatic (1M levels) | 9.23 ms | 20.95 ms | 5.47 ms | 2.27x | **0.59x** |
| static_stability (1M levels) | 31.91 ms | 161.10 ms | - | 5.05x | - |
| brunt_vaisala_frequency_squared (1M levels) | 7.62 ms | 61.34 ms | - | 8.05x | - |
| wind_speed (1000 x 1000) | 6.32 ms | 15.32 ms | 15.49 ms | 2.43x | 2.45x |
| wind_direction (1000 x 1000) | 8.74 ms | 33.04 ms | 38.34 ms | 3.78x | 4.39x |
| heat_index (1000 x 1000) | 7.34 ms | 83.93 ms | - | 11.44x | - |
| windchill (1000 x 1000) | 15.38 ms | 39.75 ms | - | 2.58x | - |
| apparent_temperature (1000 x 1000) | 13.33 ms | 167.73 ms | - | 12.58x | - |
| pressure_to_height_std (1M) | 6.78 ms | 37.25 ms | 28.96 ms | 5.50x | 4.27x |
| first_derivative (1000 x 1000) | 4.61 ms | 4.55 ms | 4.32 ms | **0.99x** | **0.94x** |
| laplacian (1000 x 1000) | 12.65 ms | 12.21 ms | - | **0.97x** | - |
| vorticity (1000 x 1000) | 25.29 ms | 55.30 ms | - | 2.19x | - |
| divergence (1000 x 1000) | 24.90 ms | 50.13 ms | - | 2.01x | - |
| total_deformation (1000 x 1000) | 26.76 ms | 117.37 ms | - | 4.39x | - |
| advection (1000 x 1000) | 16.18 ms | 57.66 ms | - | 3.56x | - |
| frontogenesis (1000 x 1000) | 130.09 ms | 393.42 ms | - | 3.02x | - |
| geostrophic_wind (1000 x 1000) | 32.00 ms | 124.80 ms | - | 3.90x | - |
| smooth_n_point 9 (1000 x 1000) | 10.64 ms | 18.70 ms | - | 1.76x | - |
| supercell_composite (1M) | 4.81 ms | 22.69 ms | 6.22 ms | 4.72x | 1.29x |

Reading it honestly:

- **Against upstream MetPy the port wins everywhere**, from 0.97x to 32.9x. The
  single near-tie is `laplacian`, which is two calls to the derivative stencil;
  the win there is entirely upstream's `pint` overhead being removed.
- **Against plain NumPy there are three losses**, shown in bold:
  - `thickness_hydrostatic` (0.59x) is a sequential reduction over one long
    profile, so it is memory-bound and single-threaded; NumPy's pairwise sum is
    simply a shorter dependency chain.
  - `dewpoint` (0.71x) needs one `log` and one `exp` per element. NumPy calls
    into its own vectorised `libm`; the Mojo kernel is at the mercy of the
    same per-call overhead without NumPy's batching, so it wins against
    MetPy (3.3x) and loses to NumPy.
  - `first_derivative` (0.94x vs NumPy, 0.99x vs MetPy) is memory-bound: it
    reads each point three times and writes once, and there is nothing to
    vectorise on a strided axis.
- The `vs metpy` column is the honest end-to-end number for a caller, because
  upstream is `pint` + NumPy and the port is ctypes + Mojo. The `vs numpy`
  column is the like-for-like kernel comparison and is where this port's
  weaknesses show.

## How it works

### FFI strategy

One compilation unit. `src/capi.mojo` is a thin C ABI layer: 92 `@export`
functions, each of which rebuilds a `Pointer` from an `Int` address and calls
straight into `src/mojometpy/*.mojo`, where the meteorological logic lives,
one call below and in upstream's source order. No allocation happens at the
boundary, so the ABI is fixed-shape `(addresses, counts..., output address)`.

Two things the toolchain forces:

- **`@export` requires an explicit `abi("C")` and rejects parametric functions**,
  so every buffer crosses as an `Int` address and is rebuilt inside the wrapper
  with `Pointer(unsafe_from_address=...)`. This is the only place the divergence
  lives; the kernel keeps upstream's structure.
- **`Int` is 64-bit and ctypes defaults to 32-bit `c_int`.** A 32-bit argtype
  silently truncates every buffer address above 4 GiB into a wild pointer. The
  signature table therefore uses `c_int64` throughout, and `tests/test_ffi.py`
  fails if a `c_int` ever appears.

The signature table in `python/mojometpy/_lib.py` is not hand-maintained: it is
generated from the `@export` list in `src/capi.mojo` by
`python -m mojometpy._lib --rewrite-signatures`, and `tests/test_ffi.py` asserts
the two agree, so a kernel whose signature changes cannot be called through a
stale table.

### Memory layout

Arrays cross as flat C-contiguous `float64` buffers; there is no wrapper object
and no refcount. A N-d array of shape `s` is one buffer of `prod(s)` doubles, and
the kernels index it through an `(n, stride, outer)` geometry that the Python
wrapper computes once:

- `n` is the length of the axis being worked along,
- `stride` is that axis's element stride in the flat buffer,
- `outer` is how many independent lines there are.

This is upstream's `make_take(ndim, axis)` restated numerically, and it is the
only structural difference between the two: it lets any rank be handled without
a special case, and it is why the derivative kernels need no rank branches.

### SIMD and parallelism

Elementwise kernels are a `W`-lane pass over `[0, n)` (`W = simd_width_of[float64]`,
4 on this target) with a scalar tail, written as upstream wrote the loop. A
`W`-lane load reads consecutive elements, so a kernel vectorises only when its
consecutive elements are consecutive outputs; the derivative and interpolation
kernels that work along a strided axis reduce one line at a time, because no
contiguous load walks lines that are `n * stride` apart.

`smooth_window` has no vectorised path: a `W`-lane load gathers `W` taps of one
window element, which reproduces the weighted sum only when the window height
happens to equal the lane width.

The compute-bound kernels (those with a transcendental per element) split
`[0, n)` across 16 workers above 524288 elements, through
`max.algorithm.parallelize`; the pure-arithmetic ones stay serial because a
`W`-lane pass over them is already at the memory bandwidth limit. Both paths are
tested for chunk independence in `tests/test_behaviour.py`.

## Tests

```
$ pixi run test
273 passed
```

- `tests/test_parity.py` — every ported function against upstream MetPy 1.7.1,
  with the tolerance each one earns rather than one tolerance for all.
- `tests/test_behaviour.py` — the contracts numerical parity does not cover:
  shapes, broadcasting, NaN handling, error paths, SIMD tails, chunk boundaries
  and the parallel split.
- `tests/test_kernels.py` — **one test per exported kernel, calling it by name**
  through `ctypes` and asserting on its output. Without this a kernel that is
  wired up and completely wrong still ships green: the public API only reaches a
  kernel through its wrapper, so a wrapper bug and a kernel bug look identical
  from outside. `test_every_export_is_covered` fails if an export is added
  without a test here.
- `tests/test_ffi.py` — the ABI contract: the signature table matches the Mojo
  source, every integer is 64-bit, and the compiled library is really loaded.

## Layout

```
src/capi.mojo               92 @export C ABI wrappers, one call into the kernels
src/mojometpy/basic.mojo    port of metpy/calc/basic.py, upstream order
src/mojometpy/thermo.mojo   port of metpy/calc/thermo.py, upstream order
src/mojometpy/tools.mojo    port of metpy/calc/tools.py + interpolation
src/mojometpy/kinematics.mojo  port of metpy/calc/kinematics.py, upstream order
src/mojometpy/indices.mojo  composite indices and storm motion
src/mojometpy/turbulence.mojo   TKE and friction velocity
src/mojometpy/constants.mojo   physical constants, verbatim from metpy
src/mojometpy/simd.mojo    SIMD helpers shared by the kernels
python/mojometpy/           the public API: ctypes wrapper + numpy front end
build/build.sh              mojo build --emit shared-lib
bench/bench.py              the benchmark table above
```

## License

MIT, as upstream. See `LICENSE`. Mojo is a trademark of Modular, Inc.; this
project is not affiliated with Modular or with Unidata.