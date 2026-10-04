"""The C ABI the Python bindings call through.

Buffers cross as `Int` addresses: `@export` refuses parametric functions, and a
pointer with an inferred origin is parametric. Nothing here allocates; the
upstream logic stays visible in `src/mojometpy/*.mojo`, one call below.
"""

from mojometpy import basic, indices, kinematics, thermo, tools, turbulence
from mojometpy.simd import W
from mojometpy.types import Ptr

# ---- thermo ----

@export("mm_moist_air_gas_constant")
def mm_moist_air_gas_constant(q: Int, n: Int, o: Int) abi("C"):
    thermo.moist_air_gas_constant(Ptr(unsafe_from_address=q), n, Ptr(unsafe_from_address=o))


@export("mm_moist_air_specific_heat_pressure")
def mm_moist_air_specific_heat_pressure(q: Int, n: Int, o: Int) abi("C"):
    thermo.moist_air_specific_heat_pressure(
        Ptr(unsafe_from_address=q), n, Ptr(unsafe_from_address=o)
    )


@export("mm_moist_air_poisson_exponent")
def mm_moist_air_poisson_exponent(q: Int, n: Int, o: Int) abi("C"):
    thermo.moist_air_poisson_exponent(Ptr(unsafe_from_address=q), n, Ptr(unsafe_from_address=o))


@export("mm_water_latent_heat_vaporization")
def mm_water_latent_heat_vaporization(t: Int, n: Int, o: Int) abi("C"):
    thermo.water_latent_heat_vaporization(Ptr(unsafe_from_address=t), n, Ptr(unsafe_from_address=o))


@export("mm_water_latent_heat_sublimation")
def mm_water_latent_heat_sublimation(t: Int, n: Int, o: Int) abi("C"):
    thermo.water_latent_heat_sublimation(Ptr(unsafe_from_address=t), n, Ptr(unsafe_from_address=o))


@export("mm_water_latent_heat_melting")
def mm_water_latent_heat_melting(t: Int, n: Int, o: Int) abi("C"):
    thermo.water_latent_heat_melting(Ptr(unsafe_from_address=t), n, Ptr(unsafe_from_address=o))


@export("mm_relative_humidity_from_dewpoint")
def mm_relative_humidity_from_dewpoint(t: Int, td: Int, phase: Int, n: Int, o: Int) abi("C"):
    thermo.relative_humidity_from_dewpoint(
        Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=td), phase, n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_exner_function")
def mm_exner_function(p: Int, p0v: Float64, n: Int, o: Int) abi("C"):
    thermo.exner_function(Ptr(unsafe_from_address=p), p0v, n, Ptr(unsafe_from_address=o))


@export("mm_potential_temperature")
def mm_potential_temperature(p: Int, t: Int, n: Int, o: Int) abi("C"):
    thermo.potential_temperature(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), n, Ptr(unsafe_from_address=o)
    )


@export("mm_temperature_from_potential_temperature")
def mm_temperature_from_potential_temperature(p: Int, th: Int, n: Int, o: Int) abi("C"):
    thermo.temperature_from_potential_temperature(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=th), n, Ptr(unsafe_from_address=o)
    )


@export("mm_dry_lapse")
def mm_dry_lapse(p: Int, t: Int, pref: Int, n: Int, o: Int) abi("C"):
    thermo.dry_lapse(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t),
        Ptr(unsafe_from_address=pref), n, Ptr(unsafe_from_address=o)
    )


@export("mm_lcl")
def mm_lcl(p: Int, t: Int, td: Int, n: Int, op: Int, ot: Int) abi("C"):
    thermo.lcl(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=td),
        n, Ptr(unsafe_from_address=op), Ptr(unsafe_from_address=ot)
    )


@export("mm_vapor_pressure")
def mm_vapor_pressure(p: Int, w: Int, n: Int, o: Int) abi("C"):
    thermo.vapor_pressure(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=w), n, Ptr(unsafe_from_address=o)
    )


@export("mm_saturation_vapor_pressure_liquid")
def mm_saturation_vapor_pressure_liquid(t: Int, n: Int, o: Int) abi("C"):
    thermo._saturation_vapor_pressure_liquid(Ptr(unsafe_from_address=t), n, Ptr(unsafe_from_address=o))


@export("mm_saturation_vapor_pressure_solid")
def mm_saturation_vapor_pressure_solid(t: Int, n: Int, o: Int) abi("C"):
    thermo._saturation_vapor_pressure_solid(Ptr(unsafe_from_address=t), n, Ptr(unsafe_from_address=o))


@export("mm_saturation_vapor_pressure")
def mm_saturation_vapor_pressure(t: Int, phase: Int, n: Int, o: Int) abi("C"):
    thermo.saturation_vapor_pressure(
        Ptr(unsafe_from_address=t), phase, n, Ptr(unsafe_from_address=o)
    )


@export("mm_dewpoint")
def mm_dewpoint(e: Int, n: Int, o: Int) abi("C"):
    thermo.dewpoint(Ptr(unsafe_from_address=e), n, Ptr(unsafe_from_address=o))


@export("mm_dewpoint_from_relative_humidity")
def mm_dewpoint_from_relative_humidity(t: Int, rh: Int, n: Int, o: Int) abi("C"):
    thermo.dewpoint_from_relative_humidity(
        Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=rh), n, Ptr(unsafe_from_address=o)
    )


@export("mm_mixing_ratio")
def mm_mixing_ratio(e: Int, p: Int, eps: Float64, n: Int, o: Int) abi("C"):
    thermo.mixing_ratio(
        Ptr(unsafe_from_address=e), Ptr(unsafe_from_address=p), eps, n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_saturation_mixing_ratio")
def mm_saturation_mixing_ratio(p: Int, t: Int, phase: Int, n: Int, o: Int) abi("C"):
    thermo.saturation_mixing_ratio(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), phase, n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_equivalent_potential_temperature")
def mm_equivalent_potential_temperature(p: Int, t: Int, td: Int, n: Int, o: Int) abi("C"):
    thermo.equivalent_potential_temperature(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=td),
        n, Ptr(unsafe_from_address=o)
    )


@export("mm_virtual_temperature")
def mm_virtual_temperature(t: Int, w: Int, eps: Float64, n: Int, o: Int) abi("C"):
    thermo.virtual_temperature(
        Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=w), eps, n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_virtual_temperature_from_dewpoint")
def mm_virtual_temperature_from_dewpoint(
    p: Int, t: Int, td: Int, eps: Float64, phase: Int, n: Int, o: Int
) abi("C"):
    thermo.virtual_temperature_from_dewpoint(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=td),
        eps, phase, n, Ptr(unsafe_from_address=o)
    )


@export("mm_virtual_potential_temperature")
def mm_virtual_potential_temperature(
    p: Int, t: Int, w: Int, eps: Float64, n: Int, o: Int
) abi("C"):
    thermo.virtual_potential_temperature(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=w),
        eps, n, Ptr(unsafe_from_address=o)
    )


@export("mm_density")
def mm_density(p: Int, t: Int, w: Int, eps: Float64, n: Int, o: Int) abi("C"):
    thermo.density(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=w),
        eps, n, Ptr(unsafe_from_address=o)
    )


@export("mm_psychrometric_vapor_pressure_wet")
def mm_psychrometric_vapor_pressure_wet(
    p: Int, tdry: Int, twet: Int, c: Float64, n: Int, o: Int
) abi("C"):
    thermo.psychrometric_vapor_pressure_wet(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=tdry),
        Ptr(unsafe_from_address=twet), c, n, Ptr(unsafe_from_address=o)
    )


@export("mm_relative_humidity_wet_psychrometric")
def mm_relative_humidity_wet_psychrometric(
    p: Int, tdry: Int, twet: Int, c: Float64, n: Int, o: Int
) abi("C"):
    thermo.relative_humidity_wet_psychrometric(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=tdry),
        Ptr(unsafe_from_address=twet), c, n, Ptr(unsafe_from_address=o)
    )


@export("mm_mixing_ratio_from_relative_humidity")
def mm_mixing_ratio_from_relative_humidity(
    p: Int, t: Int, rh: Int, phase: Int, n: Int, o: Int
) abi("C"):
    thermo.mixing_ratio_from_relative_humidity(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=rh),
        phase, n, Ptr(unsafe_from_address=o)
    )


@export("mm_relative_humidity_from_mixing_ratio")
def mm_relative_humidity_from_mixing_ratio(
    p: Int, t: Int, w: Int, phase: Int, n: Int, o: Int
) abi("C"):
    thermo.relative_humidity_from_mixing_ratio(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=w),
        phase, n, Ptr(unsafe_from_address=o)
    )


@export("mm_mixing_ratio_from_specific_humidity")
def mm_mixing_ratio_from_specific_humidity(q: Int, n: Int, o: Int) abi("C"):
    thermo.mixing_ratio_from_specific_humidity(
        Ptr(unsafe_from_address=q), n, Ptr(unsafe_from_address=o)
    )


@export("mm_specific_humidity_from_mixing_ratio")
def mm_specific_humidity_from_mixing_ratio(w: Int, n: Int, o: Int) abi("C"):
    thermo.specific_humidity_from_mixing_ratio(
        Ptr(unsafe_from_address=w), n, Ptr(unsafe_from_address=o)
    )


@export("mm_relative_humidity_from_specific_humidity")
def mm_relative_humidity_from_specific_humidity(
    p: Int, t: Int, q: Int, phase: Int, n: Int, o: Int
) abi("C"):
    thermo.relative_humidity_from_specific_humidity(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=q),
        phase, n, Ptr(unsafe_from_address=o)
    )


@export("mm_dry_static_energy")
def mm_dry_static_energy(z: Int, t: Int, n: Int, o: Int) abi("C"):
    thermo.dry_static_energy(
        Ptr(unsafe_from_address=z), Ptr(unsafe_from_address=t), n, Ptr(unsafe_from_address=o)
    )


@export("mm_moist_static_energy")
def mm_moist_static_energy(z: Int, t: Int, q: Int, n: Int, o: Int) abi("C"):
    thermo.moist_static_energy(
        Ptr(unsafe_from_address=z), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=q),
        n, Ptr(unsafe_from_address=o)
    )


@export("mm_thickness_hydrostatic")
def mm_thickness_hydrostatic(
    p: Int, t: Int, w: Int, use_mixing: Int, eps: Float64, n: Int, o: Int
) abi("C"):
    thermo.thickness_hydrostatic(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=w),
        use_mixing, eps, n, Ptr(unsafe_from_address=o)
    )


@export("mm_thickness_hydrostatic_from_relative_humidity")
def mm_thickness_hydrostatic_from_relative_humidity(
    p: Int, t: Int, rh: Int, n: Int, o: Int
) abi("C"):
    thermo.thickness_hydrostatic_from_relative_humidity(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=rh),
        n, Ptr(unsafe_from_address=o)
    )


@export("mm_brunt_vaisala_frequency_squared")
def mm_brunt_vaisala_frequency_squared(z: Int, dtheta: Int, theta: Int, n: Int, o: Int) abi("C"):
    thermo.brunt_vaisala_frequency_squared(
        Ptr(unsafe_from_address=z), Ptr(unsafe_from_address=dtheta),
        Ptr(unsafe_from_address=theta), n, Ptr(unsafe_from_address=o)
    )


@export("mm_static_stability")
def mm_static_stability(p: Int, t: Int, dlog: Int, n: Int, o: Int) abi("C"):
    thermo.static_stability(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=dlog),
        n, Ptr(unsafe_from_address=o)
    )


@export("mm_vertical_velocity_pressure")
def mm_vertical_velocity_pressure(w: Int, p: Int, t: Int, mr: Int, n: Int, o: Int) abi("C"):
    thermo.vertical_velocity_pressure(
        Ptr(unsafe_from_address=w), Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=t),
        Ptr(unsafe_from_address=mr), n, Ptr(unsafe_from_address=o)
    )


@export("mm_specific_humidity_from_dewpoint")
def mm_specific_humidity_from_dewpoint(p: Int, td: Int, phase: Int, n: Int, o: Int) abi("C"):
    thermo.specific_humidity_from_dewpoint(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=td), phase, n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_scale_height")
def mm_scale_height(tb: Int, tt: Int, n: Int, o: Int) abi("C"):
    thermo.scale_height(
        Ptr(unsafe_from_address=tb), Ptr(unsafe_from_address=tt), n, Ptr(unsafe_from_address=o)
    )


# ---- basic ----

@export("mm_wind_speed")
def mm_wind_speed(u: Int, v: Int, n: Int, o: Int) abi("C"):
    basic.wind_speed(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v), n, Ptr(unsafe_from_address=o)
    )


@export("mm_wind_direction")
def mm_wind_direction(u: Int, v: Int, convention: Int, n: Int, o: Int) abi("C"):
    basic.wind_direction(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v), convention, n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_wind_components")
def mm_wind_components(speed: Int, direction: Int, n: Int, ou: Int, ov: Int) abi("C"):
    basic.wind_components(
        Ptr(unsafe_from_address=speed), Ptr(unsafe_from_address=direction), n,
        Ptr(unsafe_from_address=ou), Ptr(unsafe_from_address=ov)
    )


@export("mm_windchill")
def mm_windchill(
    t: Int, s: Int, face: Int, mask: Int, n: Int, o: Int, om: Int
) abi("C"):
    basic.windchill(
        Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=s), face, mask, n,
        Ptr(unsafe_from_address=o), Ptr(unsafe_from_address=om)
    )


@export("mm_heat_index")
def mm_heat_index(t: Int, rh: Int, mask: Int, n: Int, o: Int, om: Int) abi("C"):
    basic.heat_index(
        Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=rh), mask, n,
        Ptr(unsafe_from_address=o), Ptr(unsafe_from_address=om)
    )


@export("mm_apparent_temperature")
def mm_apparent_temperature(
    t: Int, rh: Int, s: Int, face: Int, mask: Int, n: Int, o: Int, om: Int
) abi("C"):
    basic.apparent_temperature(
        Ptr(unsafe_from_address=t), Ptr(unsafe_from_address=rh), Ptr(unsafe_from_address=s),
        face, mask, n, Ptr(unsafe_from_address=o), Ptr(unsafe_from_address=om)
    )


@export("mm_pressure_to_height_std")
def mm_pressure_to_height_std(p: Int, n: Int, o: Int) abi("C"):
    basic.pressure_to_height_std(Ptr(unsafe_from_address=p), n, Ptr(unsafe_from_address=o))


@export("mm_height_to_geopotential")
def mm_height_to_geopotential(z: Int, n: Int, o: Int) abi("C"):
    basic.height_to_geopotential(Ptr(unsafe_from_address=z), n, Ptr(unsafe_from_address=o))


@export("mm_geopotential_to_height")
def mm_geopotential_to_height(z: Int, n: Int, o: Int) abi("C"):
    basic.geopotential_to_height(Ptr(unsafe_from_address=z), n, Ptr(unsafe_from_address=o))


@export("mm_height_to_pressure_std")
def mm_height_to_pressure_std(z: Int, n: Int, o: Int) abi("C"):
    basic.height_to_pressure_std(Ptr(unsafe_from_address=z), n, Ptr(unsafe_from_address=o))


@export("mm_coriolis_parameter")
def mm_coriolis_parameter(lat: Int, n: Int, o: Int) abi("C"):
    basic.coriolis_parameter(Ptr(unsafe_from_address=lat), n, Ptr(unsafe_from_address=o))


@export("mm_add_height_to_pressure")
def mm_add_height_to_pressure(p: Int, z: Int, n: Int, o: Int) abi("C"):
    basic.add_height_to_pressure(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=z), n, Ptr(unsafe_from_address=o)
    )


@export("mm_add_pressure_to_height")
def mm_add_pressure_to_height(z: Int, p: Int, n: Int, o: Int) abi("C"):
    basic.add_pressure_to_height(
        Ptr(unsafe_from_address=z), Ptr(unsafe_from_address=p), n, Ptr(unsafe_from_address=o)
    )


@export("mm_sigma_to_pressure")
def mm_sigma_to_pressure(sig: Int, psfc: Int, ptop: Int, n: Int, o: Int) abi("C"):
    basic.sigma_to_pressure(
        Ptr(unsafe_from_address=sig), Ptr(unsafe_from_address=psfc),
        Ptr(unsafe_from_address=ptop), n, Ptr(unsafe_from_address=o)
    )


@export("mm_altimeter_to_station_pressure")
def mm_altimeter_to_station_pressure(a: Int, z: Int, n: Int, o: Int) abi("C"):
    basic.altimeter_to_station_pressure(
        Ptr(unsafe_from_address=a), Ptr(unsafe_from_address=z), n, Ptr(unsafe_from_address=o)
    )


@export("mm_altimeter_to_sea_level_pressure")
def mm_altimeter_to_sea_level_pressure(a: Int, z: Int, t: Int, n: Int, o: Int) abi("C"):
    basic.altimeter_to_sea_level_pressure(
        Ptr(unsafe_from_address=a), Ptr(unsafe_from_address=z), Ptr(unsafe_from_address=t),
        n, Ptr(unsafe_from_address=o)
    )


# ---- tools ----

@export("mm_first_derivative")
def mm_first_derivative(
    f: Int, delta: Int, n: Int, stride: Int, outer: Int, total: Int, o: Int
) abi("C"):
    tools.first_derivative(
        Ptr(unsafe_from_address=f), Ptr(unsafe_from_address=delta), n, stride, outer,
        Ptr(unsafe_from_address=o)
    )


@export("mm_second_derivative")
def mm_second_derivative(
    f: Int, delta: Int, n: Int, stride: Int, outer: Int, total: Int, o: Int
) abi("C"):
    tools.second_derivative(
        Ptr(unsafe_from_address=f), Ptr(unsafe_from_address=delta), n, stride, outer,
        Ptr(unsafe_from_address=o)
    )


@export("mm_smooth_window")
def mm_smooth_window(
    data: Int, dst: Int, weights: Int, wy: Int, wx: Int, nrows: Int, ncols: Int, passes: Int
) abi("C"):
    tools.smooth_window(
        Ptr(unsafe_from_address=data), Ptr(unsafe_from_address=dst),
        Ptr(unsafe_from_address=weights), wy, wx, nrows, ncols, passes
    )


@export("mm_normalize_window")
def mm_normalize_window(window: Int, m: Int) abi("C"):
    tools.normalize_window(Ptr(unsafe_from_address=window), m)


@export("mm_interpolate_1d")
def mm_interpolate_1d(
    x: Int, xp: Int, values: Int, m: Int, n: Int, stride: Int, outer: Int,
    fill: Float64, o: Int,
) abi("C"):
    tools.interpolate_1d(
        Ptr(unsafe_from_address=x), Ptr(unsafe_from_address=xp),
        Ptr(unsafe_from_address=values), m, n, stride, outer, fill,
        Ptr(unsafe_from_address=o)
    )


@export("mm_resample_nn_1d_nearest")
def mm_resample_nn_1d_nearest(a: Int, center: Float64, n: Int) abi("C") -> Int:
    return tools.resample_nn_1d_nearest(Ptr(unsafe_from_address=a), center, n)


@export("mm_nearest_intersection_idx")
def mm_nearest_intersection_idx(a: Int, b: Int, n: Int, o: Int) abi("C") -> Int:
    return tools.nearest_intersection_idx(
        Ptr(unsafe_from_address=a), Ptr(unsafe_from_address=b), n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_lat_lon_grid_deltas_dx")
def mm_lat_lon_grid_deltas_dx(
    lon: Int, lat: Int, n: Int, stride: Int, outer: Int, o: Int
) abi("C"):
    tools.lat_lon_grid_deltas_dx(
        Ptr(unsafe_from_address=lon), Ptr(unsafe_from_address=lat), n, stride, outer,
        Ptr(unsafe_from_address=o)
    )


@export("mm_lat_lon_grid_deltas_dy")
def mm_lat_lon_grid_deltas_dy(
    lon: Int, lat: Int, n: Int, stride: Int, outer: Int, o: Int
) abi("C"):
    tools.lat_lon_grid_deltas_dy(
        Ptr(unsafe_from_address=lon), Ptr(unsafe_from_address=lat), n, stride, outer,
        Ptr(unsafe_from_address=o)
    )


# ---- kinematics ----

@export("mm_vector_derivative")
def mm_vector_derivative(
    u: Int, v: Int, dx: Int, dy: Int,
    nx: Int, x_stride: Int, x_outer: Int,
    ny: Int, y_stride: Int, y_outer: Int,
    dudx: Int, dudy: Int, dvdx: Int, dvdy: Int,
) abi("C"):
    kinematics.vector_derivative(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v),
        Ptr(unsafe_from_address=dx), Ptr(unsafe_from_address=dy),
        nx, x_stride, x_outer, ny, y_stride, y_outer,
        Ptr(unsafe_from_address=dudx), Ptr(unsafe_from_address=dudy),
        Ptr(unsafe_from_address=dvdx), Ptr(unsafe_from_address=dvdy)
    )


@export("mm_vector_derivative_map_correction")
def mm_vector_derivative_map_correction(
    u: Int, v: Int, dpdy: Int, dmdx: Int, ps: Int, ms: Int, n: Int,
    dudx: Int, dudy: Int, dvdx: Int, dvdy: Int,
) abi("C"):
    kinematics.vector_derivative_map_correction(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v),
        Ptr(unsafe_from_address=dpdy), Ptr(unsafe_from_address=dmdx),
        Ptr(unsafe_from_address=ps), Ptr(unsafe_from_address=ms), n,
        Ptr(unsafe_from_address=dudx), Ptr(unsafe_from_address=dudy),
        Ptr(unsafe_from_address=dvdx), Ptr(unsafe_from_address=dvdy)
    )


@export("mm_vorticity")
def mm_vorticity(dudy: Int, dvdx: Int, n: Int, o: Int) abi("C"):
    kinematics.vorticity(
        Ptr(unsafe_from_address=dudy), Ptr(unsafe_from_address=dvdx), n, Ptr(unsafe_from_address=o)
    )


@export("mm_curvature_vorticity")
def mm_curvature_vorticity(
    u: Int, v: Int, dudx: Int, dudy: Int, dvdx: Int, dvdy: Int, n: Int, o: Int
) abi("C"):
    kinematics.curvature_vorticity(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v),
        Ptr(unsafe_from_address=dudx), Ptr(unsafe_from_address=dudy),
        Ptr(unsafe_from_address=dvdx), Ptr(unsafe_from_address=dvdy), n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_shear_vorticity")
def mm_shear_vorticity(
    u: Int, v: Int, dudx: Int, dudy: Int, dvdx: Int, dvdy: Int, n: Int, o: Int
) abi("C"):
    kinematics.shear_vorticity(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v),
        Ptr(unsafe_from_address=dudx), Ptr(unsafe_from_address=dudy),
        Ptr(unsafe_from_address=dvdx), Ptr(unsafe_from_address=dvdy), n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_divergence")
def mm_divergence(dudx: Int, dvdy: Int, n: Int, o: Int) abi("C"):
    kinematics.divergence(
        Ptr(unsafe_from_address=dudx), Ptr(unsafe_from_address=dvdy), n, Ptr(unsafe_from_address=o)
    )


@export("mm_shearing_deformation")
def mm_shearing_deformation(dudy: Int, dvdx: Int, n: Int, o: Int) abi("C"):
    kinematics.shearing_deformation(
        Ptr(unsafe_from_address=dudy), Ptr(unsafe_from_address=dvdx), n, Ptr(unsafe_from_address=o)
    )


@export("mm_stretching_deformation")
def mm_stretching_deformation(dudx: Int, dvdy: Int, n: Int, o: Int) abi("C"):
    kinematics.stretching_deformation(
        Ptr(unsafe_from_address=dudx), Ptr(unsafe_from_address=dvdy), n, Ptr(unsafe_from_address=o)
    )


@export("mm_total_deformation")
def mm_total_deformation(dudx: Int, dudy: Int, dvdx: Int, dvdy: Int, n: Int, o: Int) abi("C"):
    kinematics.total_deformation(
        Ptr(unsafe_from_address=dudx), Ptr(unsafe_from_address=dudy),
        Ptr(unsafe_from_address=dvdx), Ptr(unsafe_from_address=dvdy), n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_advection")
def mm_advection(
    dfdx: Int, dfdy: Int, dfdz: Int, u: Int, v: Int, w: Int, use_w: Int, n: Int, o: Int
) abi("C"):
    kinematics.advection(
        Ptr(unsafe_from_address=dfdx), Ptr(unsafe_from_address=dfdy),
        Ptr(unsafe_from_address=dfdz), Ptr(unsafe_from_address=u),
        Ptr(unsafe_from_address=v), Ptr(unsafe_from_address=w), use_w, n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_frontogenesis")
def mm_frontogenesis(
    ddx_theta: Int, ddy_theta: Int, shrd: Int, strd: Int, tdef: Int, div: Int, n: Int, o: Int
) abi("C"):
    kinematics.frontogenesis(
        Ptr(unsafe_from_address=ddx_theta), Ptr(unsafe_from_address=ddy_theta),
        Ptr(unsafe_from_address=shrd), Ptr(unsafe_from_address=strd),
        Ptr(unsafe_from_address=tdef), Ptr(unsafe_from_address=div), n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_geostrophic_wind")
def mm_geostrophic_wind(dhdx: Int, dhdy: Int, lat: Int, n: Int, ou: Int, ov: Int) abi("C"):
    kinematics.geostrophic_wind(
        Ptr(unsafe_from_address=dhdx), Ptr(unsafe_from_address=dhdy),
        Ptr(unsafe_from_address=lat), n, Ptr(unsafe_from_address=ou),
        Ptr(unsafe_from_address=ov)
    )


@export("mm_montgomery_streamfunction")
def mm_montgomery_streamfunction(z: Int, t: Int, n: Int, o: Int) abi("C"):
    kinematics.montgomery_streamfunction(
        Ptr(unsafe_from_address=z), Ptr(unsafe_from_address=t), n, Ptr(unsafe_from_address=o)
    )


@export("mm_storm_relative_helicity")
def mm_storm_relative_helicity(
    u: Int, v: Int, storm_u: Float64, storm_v: Float64, n: Int, o: Int
) abi("C"):
    kinematics.storm_relative_helicity(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v),
        storm_u, storm_v, n, Ptr(unsafe_from_address=o)
    )


@export("mm_potential_vorticity_baroclinic")
def mm_potential_vorticity_baroclinic(
    dudp: Int, dthetady: Int, dvdp: Int, dthetadx: Int, avor: Int, dthetadp: Int,
    n: Int, o: Int,
) abi("C"):
    kinematics.potential_vorticity_baroclinic(
        Ptr(unsafe_from_address=dudp), Ptr(unsafe_from_address=dthetady),
        Ptr(unsafe_from_address=dvdp), Ptr(unsafe_from_address=dthetadx),
        Ptr(unsafe_from_address=avor), Ptr(unsafe_from_address=dthetadp), n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_potential_vorticity_barotropic")
def mm_potential_vorticity_barotropic(avor: Int, height: Int, n: Int, o: Int) abi("C"):
    kinematics.potential_vorticity_barotropic(
        Ptr(unsafe_from_address=avor), Ptr(unsafe_from_address=height), n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_inertial_advective_wind")
def mm_inertial_advective_wind(
    u: Int, v: Int, dugdx: Int, dugdy: Int, dvgdx: Int, dvgdy: Int, lat: Int,
    n: Int, ou: Int, ov: Int,
) abi("C"):
    kinematics.inertial_advective_wind(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v),
        Ptr(unsafe_from_address=dugdx), Ptr(unsafe_from_address=dugdy),
        Ptr(unsafe_from_address=dvgdx), Ptr(unsafe_from_address=dvgdy),
        Ptr(unsafe_from_address=lat), n, Ptr(unsafe_from_address=ou),
        Ptr(unsafe_from_address=ov)
    )


@export("mm_q_vector")
def mm_q_vector(
    dudx: Int, dudy: Int, dvdx: Int, dvdy: Int, dtempdx: Int, dtempdy: Int,
    pressure: Int, stability: Float64, n: Int, oq1: Int, oq2: Int,
) abi("C"):
    kinematics.q_vector(
        Ptr(unsafe_from_address=dudx), Ptr(unsafe_from_address=dudy),
        Ptr(unsafe_from_address=dvdx), Ptr(unsafe_from_address=dvdy),
        Ptr(unsafe_from_address=dtempdx), Ptr(unsafe_from_address=dtempdy),
        Ptr(unsafe_from_address=pressure), stability, n,
        Ptr(unsafe_from_address=oq1), Ptr(unsafe_from_address=oq2)
    )


# ---- turbulence ----

@export("mm_tke")
def mm_tke(u: Int, v: Int, w: Int, perturbation: Int, n: Int, stride: Int, outer: Int, o: Int) abi("C"):
    turbulence.tke(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=v),
        Ptr(unsafe_from_address=w), perturbation, n, stride, outer,
        Ptr(unsafe_from_address=o)
    )


@export("mm_friction_velocity")
def mm_friction_velocity(
    u: Int, w: Int, v: Int, has_v: Int, perturbation: Int, n: Int, stride: Int,
    outer: Int, o: Int,
) abi("C"):
    turbulence.friction_velocity(
        Ptr(unsafe_from_address=u), Ptr(unsafe_from_address=w),
        Ptr(unsafe_from_address=v), has_v, perturbation, n, stride, outer,
        Ptr(unsafe_from_address=o)
    )


# ---- indices ----

@export("mm_supercell_composite")
def mm_supercell_composite(mucape: Int, esh: Int, eshear: Int, n: Int, o: Int) abi("C"):
    indices.supercell_composite(
        Ptr(unsafe_from_address=mucape), Ptr(unsafe_from_address=esh),
        Ptr(unsafe_from_address=eshear), n, Ptr(unsafe_from_address=o)
    )


@export("mm_significant_tornado")
def mm_significant_tornado(
    sbcape: Int, lcl: Int, sh: Int, shear: Int, n: Int, o: Int
) abi("C"):
    indices.significant_tornado(
        Ptr(unsafe_from_address=sbcape), Ptr(unsafe_from_address=lcl),
        Ptr(unsafe_from_address=sh), Ptr(unsafe_from_address=shear), n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_brunt_vaisala_frequency")
def mm_brunt_vaisala_frequency(bv: Int, n: Int, o: Int) abi("C"):
    thermo.brunt_vaisala_frequency(Ptr(unsafe_from_address=bv), n, Ptr(unsafe_from_address=o))


@export("mm_brunt_vaisala_period")
def mm_brunt_vaisala_period(bv: Int, n: Int, o: Int) abi("C"):
    thermo.brunt_vaisala_period(Ptr(unsafe_from_address=bv), n, Ptr(unsafe_from_address=o))


@export("mm_gradient_richardson_number")
def mm_gradient_richardson_number(
    dthetadz: Int, dudz: Int, dvdz: Int, theta: Int, n: Int, o: Int
) abi("C"):
    thermo.gradient_richardson_number(
        Ptr(unsafe_from_address=dthetadz), Ptr(unsafe_from_address=dudz),
        Ptr(unsafe_from_address=dvdz), Ptr(unsafe_from_address=theta), n,
        Ptr(unsafe_from_address=o)
    )


@export("mm_dewpoint_from_specific_humidity")
def mm_dewpoint_from_specific_humidity(p: Int, q: Int, n: Int, o: Int) abi("C"):
    thermo.dewpoint_from_specific_humidity(
        Ptr(unsafe_from_address=p), Ptr(unsafe_from_address=q), n, Ptr(unsafe_from_address=o)
    )


