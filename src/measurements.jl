# =============================================================================
# Measurement models z = h(x) + v, v ~ N(0, R)
# =============================================================================
#
# Measurements are taken from a fixed origin in the dynamics model's own frame:
#   - the geocentre for the inertial Kepler/J2 problems (origin = 0), or
#   - the Earth's position (-μ, 0, 0) in the CR3BP rotating frame.
# Units follow the model: km and km/s, or nondimensional.
#
# Range-rate is the time derivative of the distance to the origin. Because the
# origin is fixed in the frame, it is the same in the rotating and the inertial
# frame: ω × (r - r_E) is perpendicular to r - r_E.

"""
    MeasurementModel

Abstract type for `z = h(x) + v`. Subtypes implement [`measure`](@ref) and
[`measurement_cov`](@ref).
"""
abstract type MeasurementModel end

"""
    RangeRangeRate(origin, σρ, σρ̇)

Range `|r - origin|` and range-rate `(r - origin)·v / |r - origin|`. The
standard deviations are in the model's length and speed units.
"""
struct RangeRangeRate <: MeasurementModel
    origin::SVector{3,Float64}
    σρ::Float64
    σρ̇::Float64
end

"""
    Range(origin, σ)

Range `|r - origin|` alone, with standard deviation `σ` in the model's length unit.
"""
struct Range <: MeasurementModel
    origin::SVector{3,Float64}
    σ::Float64
end

"""
    RangeRate(origin, σ)

Range-rate `(r - origin)·v / |r - origin|` alone, with standard deviation `σ` in
the model's speed unit.
"""
struct RangeRate <: MeasurementModel
    origin::SVector{3,Float64}
    σ::Float64
end

"""
    Angles(origin, σ)

Azimuth `atan(Δy, Δx)` and elevation `asin(Δz / |Δr|)` of `Δr = r - origin`,
in radians, with standard deviation `σ` [rad] on each.

The azimuth has a branch cut at ±π. The derivatives are local, so the
measurement-space criterion is unaffected. Anything that uses samples of the
azimuth (residuals, moments, likelihoods) must wrap differences into (-π, π].
"""
struct Angles <: MeasurementModel
    origin::SVector{3,Float64}
    σ::Float64
end

"""
    measure(meas, x) -> SVector

Noise-free measurement `h(x)`. It is generic in the element type, so it accepts
dual numbers for derivatives.
"""
function measure end

function measure(m::RangeRangeRate, x::AbstractVector)
    d = SVector(x[1] - m.origin[1], x[2] - m.origin[2], x[3] - m.origin[3])
    ρ = sqrt(d[1]^2 + d[2]^2 + d[3]^2)
    return SVector(ρ, (d[1] * x[4] + d[2] * x[5] + d[3] * x[6]) / ρ)
end

measure(m::Range, x::AbstractVector) = SVector(measure(RangeRangeRate(m.origin, 1.0, 1.0), x)[1])
measure(m::RangeRate, x::AbstractVector) = SVector(measure(RangeRangeRate(m.origin, 1.0, 1.0), x)[2])

function measure(m::Angles, x::AbstractVector)
    d = SVector(x[1] - m.origin[1], x[2] - m.origin[2], x[3] - m.origin[3])
    ρ = sqrt(d[1]^2 + d[2]^2 + d[3]^2)
    return SVector(atan(d[2], d[1]), asin(d[3] / ρ))
end

"""
    measurement_cov(meas) -> SMatrix

Noise covariance `R`.
"""
measurement_cov(m::RangeRangeRate) = SMatrix{2,2,Float64}(m.σρ^2, 0, 0, m.σρ̇^2)
measurement_cov(m::Angles) = SMatrix{2,2,Float64}(m.σ^2, 0, 0, m.σ^2)
measurement_cov(m::Union{Range,RangeRate}) = SMatrix{1,1,Float64}(m.σ^2)

"""
    measurement_jacobian(meas, x) -> Matrix

`H = ∂h/∂x` at `x`, from ForwardDiff.
"""
measurement_jacobian(m::MeasurementModel, x::AbstractVector) =
    ForwardDiff.jacobian(y -> measure(m, y), SVector{6,Float64}(x))

"""
    measurement_hessian(meas, x) -> Array{Float64,3}

`H2[i, l, k] = ∂²hᵢ/∂xₗ∂xₖ` at `x`, from ForwardDiff and symmetrized in `(l, k)`.
"""
function measurement_hessian(m::MeasurementModel, x::AbstractVector)
    xs = SVector{6,Float64}(x)
    nz = length(measure(m, xs))
    H = reshape(ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(z -> measure(m, z), y)), xs), nz, 6, 6)
    return (H + permutedims(H, (1, 3, 2))) / 2
end
