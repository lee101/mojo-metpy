"""mojo-metpy: MetPy's meteorological calculations, implemented in Mojo.

The public names, argument order and defaults are upstream MetPy's.  What is
not ported is the unit layer: arguments and results are plain float64 arrays
in SI base units, so `mojometpy.thermo.potential_temperature(85000.0, 288.0)`
is the same number as `metpy.calc.potential_temperature(850 * units.hPa,
288 * units.kelvin).magnitude`.

`mojometpy.calc` mirrors `metpy.calc`: the same five modules, the same
function names.
"""

from . import basic, calc, constants, indices, kinematics, thermo, tools, turbulence

__version__ = "0.1.0"
