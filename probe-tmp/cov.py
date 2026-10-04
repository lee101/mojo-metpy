"""Which public API functions have a real upstream parity case, and which do not?"""
import inspect
import re
import sys

sys.path.insert(0, "probe-tmp")
sys.path.insert(0, "tests")
import numpy as np

import test_parity as TP

MODS = ["basic", "thermo", "tools", "kinematics", "indices", "turbulence"]
parity_names = {c[0] for c in TP.CASES}
# normalise a parity case name to a function name
def norm(s):
    s = re.sub(r"_.*$", "", s)
    return s

parity_bases = {norm(n) for n in parity_names}

rows = []
for m in MODS:
    mod = __import__(f"mojometpy.{m}", fromlist=["*"])
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if fn.__module__ != f"mojometpy.{m}":
            continue
        rows.append((m, name, name in parity_names or name in parity_bases))

total = len(rows)
cov = sum(1 for r in rows if r[2])
print(f"public functions in the six calc modules: {total}")
print(f"with a named parity case: {cov}   without: {total - cov}")
print()
cur = None
for m, name, ok in sorted(rows):
    if m != cur:
        print(f"--- {m} ---")
        cur = m
    if not ok:
        print(f"   NO PARITY: {name}")
