"""Port of `metpy/calc/indices.py` (MetPy 1.7.1).

Only the composite indices that are pure elementwise arithmetic are ported;
the layer-based ones (`precipitable_water`, `bulk_shear`, `bunkers_storm_motion`,
`critical_angle`) depend on `get_layer`, which is not part of this port.
"""

from std.math import max, min
from mojometpy.simd import F, W, flag
from mojometpy.types import Ptr


def supercell_composite(
    mucape: Ptr,
    effective_storm_helicity: Ptr,
    effective_shear: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        var raw = effective_shear.unsafe_load[width=W](i)
        # upstream zeroes the shear below 10 rather than clamping it there
        var shear = min(max(raw, F(0.0)), F(20.0)) * flag(SIMD.ge(raw, F(10.0)))
        dst.unsafe_store(
            i,
            mucape.unsafe_load[width=W](i)
            / F(1000.0)
            * (effective_storm_helicity.unsafe_load[width=W](i) / F(50.0))
            * (shear / F(20.0)),
        )
        i += W
    while i < n:
        var effective_shear = effective_shear[unsafe_offset=i]
        if effective_shear > 20.0:
            effective_shear = 20.0
        if effective_shear < 10.0:
            effective_shear = 0.0
        effective_shear = effective_shear / 20.0
        dst[unsafe_offset=i] = (
            mucape[unsafe_offset=i] / 1000.0
            * (effective_storm_helicity[unsafe_offset=i] / 50.0)
            * effective_shear
        )
        i += 1


def significant_tornado(
    sbcape: Ptr,
    surface_based_lcl_height: Ptr,
    storm_helicity_1km: Ptr,
    shear_6km: Ptr,
    n: Int,
    dst: Ptr,
):
    var i = 0
    while i + W <= n:
        var raw = shear_6km.unsafe_load[width=W](i)
        # upstream zeroes the shear below 12.5 rather than clamping it there
        var shear = min(max(raw, F(0.0)), F(30.0)) * flag(
            SIMD.ge(raw, F(12.5))
        )
        var lcl = min(
            max(surface_based_lcl_height.unsafe_load[width=W](i), F(1000.0)), F(2000.0)
        )
        dst.unsafe_store(
            i,
            sbcape.unsafe_load[width=W](i)
            / F(1500.0)
            * ((F(2000.0) - lcl) / F(1000.0))
            * (storm_helicity_1km.unsafe_load[width=W](i) / F(150.0))
            * (shear / F(20.0)),
        )
        i += W
    while i < n:
        var lcl = surface_based_lcl_height[unsafe_offset=i]
        if lcl < 1000.0:
            lcl = 1000.0
        if lcl > 2000.0:
            lcl = 2000.0
        lcl = (2000.0 - lcl) / 1000.0
        var shear = shear_6km[unsafe_offset=i]
        if shear < 12.5:
            shear = 0.0
        if shear > 30.0:
            shear = 30.0
        shear = shear / 20.0
        dst[unsafe_offset=i] = (
            sbcape[unsafe_offset=i] / 1500.0
            * lcl
            * (storm_helicity_1km[unsafe_offset=i] / 150.0)
            * shear
        )
        i += 1

