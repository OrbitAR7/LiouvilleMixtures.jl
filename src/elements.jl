# =============================================================================
# Modified equinoctial elements
# =============================================================================
#
# (p, f, g, h, k, L) of Walker, Ireland and Owens (Celestial Mechanics 36, 1985):
#   p = a(1 - e²),  f = e cos(ω + Ω),  g = e sin(ω + Ω),
#   h = tan(i/2) cos Ω,  k = tan(i/2) sin Ω,  L = Ω + ω + ν.
# They are non-singular for circular and equatorial orbits (but not at i = π).
# Inputs are inertial Cartesian [r; v] in km and km/s, with μ in km³/s². The
# outputs are p in km and L in radians; the other elements are dimensionless.
#
# L is an angle with a branch cut at ±π. Callers that use samples of L must
# unwrap them (see `unwrap_angle`).

"""
    cartesian_to_equinoctial(x, μ) -> SVector{6}

Modified equinoctial elements `(p, f, g, h, k, L)` of the inertial state
`x = [r; v]` [km, km/s] for gravitational parameter `μ` [km³/s²]. It is generic
in the element type, so it works with ForwardDiff for Jacobians.
"""
function cartesian_to_equinoctial(x::AbstractVector, μ::Real)
    r = SVector(x[1], x[2], x[3])
    v = SVector(x[4], x[5], x[6])
    hv = cross(r, v)
    hn = sqrt(dot(hv, hv))
    p = hn^2 / μ
    den = hn + hv[3]
    h = -hv[2] / den
    k = hv[1] / den
    s² = 1 + h^2 + k^2
    f̂ = SVector(1 - k^2 + h^2, 2h * k, -2k) / s²
    ĝ = SVector(2h * k, 1 + k^2 - h^2, 2h) / s²
    ev = cross(v, hv) / μ - r / sqrt(dot(r, r))
    return SVector(p, dot(ev, f̂), dot(ev, ĝ), h, k, atan(dot(r, ĝ), dot(r, f̂)))
end

"""
    equinoctial_to_cartesian(q, μ) -> SVector{6}

Inverse of [`cartesian_to_equinoctial`](@ref): the inertial state [km, km/s]
from `q = (p, f, g, h, k, L)`.
"""
function equinoctial_to_cartesian(q::AbstractVector, μ::Real)
    p, f, g, h, k, L = q
    α² = h^2 - k^2
    s² = 1 + h^2 + k^2
    w = 1 + f * cos(L) + g * sin(L)
    r = p / w
    sL, cL = sin(L), cos(L)
    c = sqrt(μ / p) / s²
    return SVector(
        r / s² * (cL + α² * cL + 2h * k * sL),
        r / s² * (sL - α² * sL + 2h * k * cL),
        2r / s² * (h * sL - k * cL),
        -c * (sL + α² * sL - 2h * k * cL + g - 2f * h * k + α² * g),
        -c * (-cL + α² * cL + 2h * k * sL - f + 2g * h * k + α² * f),
        2c * (h * cL + k * sL + f * h + g * k))
end

"""
    equinoctial_logdetjac(x, μ)

`ln |det ∂(p, f, g, h, k, L)/∂(r, v)|` at `x`, from ForwardDiff. This is the
Jacobian term that the negentropy in equinoctial coordinates needs (I2 with a
coordinate map).
"""
equinoctial_logdetjac(x::AbstractVector, μ::Real) =
    logabsdet(ForwardDiff.jacobian(y -> cartesian_to_equinoctial(y, μ), SVector{6,Float64}(x)))[1]

"""
    unwrap_angle(θ, θref)

`θ` shifted by a multiple of 2π into `(θref - π, θref + π]`.
"""
unwrap_angle(θ::Real, θref::Real) = θref + rem2pi(θ - θref, RoundNearest)
