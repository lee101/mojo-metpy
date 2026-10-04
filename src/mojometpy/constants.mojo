"""Physical constants, verbatim from `metpy/constants/nounit.py` (MetPy 1.7.1).

All values are floats in SI base units, which is what the unitless Python API
of this port exchanges. `t0`, `p0` and `gamma` are the standard-atmosphere
constants module-level in upstream `metpy/calc/basic.py`.
"""

comptime Rd: Float64 = 287.04749097718457
comptime Rv: Float64 = 461.52311572606084
comptime Lv: Float64 = 2500840.0
comptime Lf: Float64 = 333700.0
comptime Ls: Float64 = 2834540.0
comptime Cp_d: Float64 = 1004.6662184201462
comptime Cp_l: Float64 = 4219.400000000001
comptime Cp_v: Float64 = 1860.078011865639
comptime Cp_i: Float64 = 2090.0
comptime zero_degc: Float64 = 273.15
comptime sat_pressure_0c: Float64 = 611.2
comptime epsilon: Float64 = 0.6219569100577033
comptime kappa: Float64 = 0.28571428571428564
comptime g: Float64 = 9.80665
comptime T0: Float64 = 273.16
comptime P0: Float64 = 100000.0
comptime Re: Float64 = 6371008.7714
# PROJ's `Geod(ellps='sphere')`, which `lat_lon_grid_deltas` uses by default.
comptime geod_sphere_radius: Float64 = 6370997.0
comptime omega: Float64 = 0.00007292115
comptime rho_l: Float64 = 999.97495

# metpy/calc/basic.py: constants for a standard atmosphere
comptime t0: Float64 = 288.0
comptime p0: Float64 = 101325.0
comptime gamma: Float64 = 0.0065
