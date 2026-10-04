"""Physical constants in SI base units, matching `metpy.constants.nounit`.

Upstream exposes these through `pint` quantities; the values here are the
`.m_as(...)` base-unit magnitudes, so `mojometpy.constants.Rd` and
`metpy.constants.nounit.Rd` are the same number.  `t0`, `p0` and `gamma` are
the standard-atmosphere constants that upstream keeps module-level in
`metpy/calc/basic.py`.
"""

Rd = 287.04749097718457
Rv = 461.52311572606084
Lv = 2500840.0
Lf = 333700.0
Ls = 2834540.0
Cp_d = 1004.6662184201462
Cp_l = 4219.400000000001
Cp_v = 1860.078011865639
Cp_i = 2090.0
zero_degc = 273.15
sat_pressure_0c = 611.2
epsilon = 0.6219569100577033
kappa = 0.28571428571428564
g = 9.80665
T0 = 273.16
P0 = 100000.0
Re = 6371008.7714
omega = 0.00007292115
rho_l = 999.97495

t0 = 288.0
p0 = 101325.0
gamma = 0.0065
