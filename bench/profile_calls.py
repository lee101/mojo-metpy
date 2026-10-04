"""Stage-level profile of the port: where the wall clock inside each public call goes.

Run under the same flock as the benchmark:

    flock /tmp/mojo-bench.lock python bench/profile_calls.py
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojometpy.basic as mmb  # noqa: E402
import mojometpy.calc as mmc  # noqa: E402
import mojometpy.kinematics as mmk  # noqa: E402
import mojometpy.thermo as mmt  # noqa: E402
import mojometpy.tools as mmt2  # noqa: E402

N1 = 1_000_000
N2 = 1000


def timeit(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


import math  # noqa: E402


def profile(label, fn, n=20):
    fn()
    t = timeit(fn, repeat=3)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n):
        fn()
    pr.disable()
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s).sort_stats("tottime")
    st.print_stats(8)
    print(f"=== {label}: {t*1e3:.2f} ms/call")
    print("\n".join(s.getvalue().splitlines()[4:14]))


rng = np.random.default_rng(1)
p = np.linspace(100000.0, 10000.0, N1)
t = 288.0 - 0.0065 * np.linspace(0.0, 15600.0, N1)
th = mmt.potential_temperature(p, t)
grid = 5000.0 + 200.0 * rng.uniform(-1, 1, (N2, N2))
u = rng.uniform(-20, 20, (N2, N2))
v = rng.uniform(-20, 20, (N2, N2))
delta1 = np.full(N1 - 1, 5000.0)
d5000 = np.full(N2 - 1, 5000.0)

profile("thermo.static_stability", lambda: mmt.static_stability(p, t))
profile("thermo.brunt_vaisala_frequency_squared", lambda: mmt.brunt_vaisala_frequency_squared(
    np.linspace(0, 16000, N1), 300.0 + 0.003 * np.linspace(0, 16000, N1)))
profile("tools.first_derivative axis=-1", lambda: mmt2.first_derivative(grid, axis=-1, delta=5000.0))
profile("tools.first_derivative axis=-2", lambda: mmt2.first_derivative(grid, axis=-2, delta=5000.0))
profile("tools.second_derivative axis=-1", lambda: mmt2.second_derivative(grid, axis=-1, delta=5000.0))
profile("tools.second_derivative axis=-2", lambda: mmt2.second_derivative(grid, axis=-2, delta=5000.0))
profile("tools.laplacian", lambda: mmt2.laplacian(grid, axes=(-1, -2), deltas=(5000.0, 5000.0)))
profile("kinematics.vorticity", lambda: mmk.vorticity(u, v, dx=5000.0, dy=5000.0))
profile("kinematics.divergence", lambda: mmk.divergence(u, v, dx=5000.0, dy=5000.0))
profile("kinematics.total_deformation", lambda: mmk.total_deformation(u, v, 5000.0, 5000.0))
profile("kinematics.advection", lambda: mmk.advection(grid, u, v, dx=5000.0, dy=5000.0))
profile("kinematics.geostrophic_wind", lambda: mmk.geostrophic_wind(grid, dx=5000.0, dy=5000.0,
                                                                 latitude=np.deg2rad(np.tile(
                                                                     np.linspace(-89, 89, N2), (N2, 1)))))
profile("kinematics.frontogenesis", lambda: mmk.frontogenesis(th[:N2 * N2].reshape(N2, N2), u, v,
                                                                dx=5000.0, dy=5000.0))
profile("basic.wind_speed", lambda: mmb.wind_speed(u, v))
profile("basic.pressure_to_height_std", lambda: mmb.pressure_to_height_std(
    np.linspace(20000.0, 100000.0, N1)))
profile("thermo.dewpoint", lambda: mmt.dewpoint(np.linspace(50.0, 5000.0, N1)))