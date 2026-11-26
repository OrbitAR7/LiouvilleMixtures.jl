# =============================================================================
# Dynamics models: vector field f, Jacobian A = ∂f/∂x, Hessian A2 = ∂²f/∂x²
# =============================================================================
#
# Units and frames
#   TwoBody, J2Gravity : inertial Cartesian [r; v] in km and km/s, time in s.
#   CR3BP              : rotating-frame [r; v] in the nondimensional units of
#                        OrbitalMechanics.CR3BPSystem (length l_star, time t_star,
#                        mean motion 1).
#
# Every model here has trace A = 0, so its flow preserves phase-space volume
# (Liouville) and det Φ = 1. For the two inertial models (r, v) are canonical
# coordinates (unit mass). The rotating-frame (r, v) of the CR3BP are not; see
# `symplectic_form`.
#
# Constants (GM_EARTH, R_EARTH, J2_EARTH, the Earth-Moon mass ratio) are those of
# OrbitalMechanics.jl. They are not redefined here.

"""
    DynamicsModel

Abstract type for an autonomous vector field `ẋ = f(x)` on a 6-dimensional
position-velocity state. Subtypes implement [`vector_field`](@ref). They can
override [`state_jacobian`](@ref) and [`state_hessian`](@ref) with analytic forms.
"""
abstract type DynamicsModel end

"""
    TwoBody(μ = GM_EARTH)

Keplerian point-mass gravity, `r̈ = -μ r / |r|³`. Inertial Cartesian state in km
and km/s, with `μ` in km³/s².
"""
struct TwoBody{T<:Real} <: DynamicsModel
    μ::T
end
TwoBody() = TwoBody(GM_EARTH)

"""
    J2Gravity(μ = GM_EARTH, J2 = J2_EARTH, R = R_EARTH)

Point mass plus the J2 zonal term. Inertial Cartesian state in km and km/s, with
the z axis along the body's spin axis. The acceleration is the gradient of

    U(r) = μ/r · [1 - J2 (R/r)² P₂(z/r)],   P₂(s) = (3s² - 1)/2

(the geopotential truncated at degree 2, zonal only). The test suite checks
[`vector_field`](@ref) against the gradient of [`potential`](@ref).
"""
struct J2Gravity{T<:Real} <: DynamicsModel
    μ::T
    J2::T
    R::T
end
J2Gravity() = J2Gravity(GM_EARTH, J2_EARTH, R_EARTH)

"""
    CR3BP(system::CR3BPSystem = EARTH_MOON)

Circular restricted three-body problem in the rotating frame, in the
nondimensional units of `system` (OrbitalMechanics.jl). The equations of motion
and the analytic Jacobian come from `OrbitalMechanics.cr3bp_eom` and
`OrbitalMechanics.cr3bp_jacobian`.
"""
struct CR3BP <: DynamicsModel
    system::CR3BPSystem
end
CR3BP() = CR3BP(EARTH_MOON)

"""
    vector_field(model, x) -> SVector{6}

Time derivative `f(x)` of the state `x = [r; v]`. Units and frame are those of
`model`. The function is generic in the element type, so it accepts dual
numbers (ForwardDiff) and Taylor polynomials (jet transport).
"""
function vector_field end

@inline function vector_field(m::TwoBody, x::SVector{6})
    r = SVector(x[1], x[2], x[3])
    r³ = (r[1]^2 + r[2]^2 + r[3]^2)^(3 / 2)
    a = -m.μ * r / r³
    return SVector(x[4], x[5], x[6], a[1], a[2], a[3])
end

@inline function vector_field(m::J2Gravity, x::SVector{6})
    # a = ∇U with U as in the J2Gravity docstring:
    #   a = -μ r/r³ - (3/2) J2 μ R²/r⁵ [x(1 - 5z²/r²), y(1 - 5z²/r²), z(3 - 5z²/r²)]
    r² = x[1]^2 + x[2]^2 + x[3]^2
    r = sqrt(r²)
    c = -m.μ / (r² * r)
    k = -(3 / 2) * m.J2 * m.μ * m.R^2 / (r² * r² * r)
    s = 5 * x[3]^2 / r²
    ax = c * x[1] + k * x[1] * (1 - s)
    ay = c * x[2] + k * x[2] * (1 - s)
    az = c * x[3] + k * x[3] * (3 - s)
    return SVector(x[4], x[5], x[6], ax, ay, az)
end

@inline vector_field(m::CR3BP, x::SVector{6}) = cr3bp_eom(x, m.system.μ)

vector_field(m::DynamicsModel, x::AbstractVector) = vector_field(m, SVector{6}(x))

"""
    potential(model::J2Gravity, r)

Gravitational potential `U` [km²/s²] at position `r` [km], with the sign
convention `a = +∇U`. Used to check [`vector_field`](@ref).
"""
function potential(m::J2Gravity, r::AbstractVector)
    ρ = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
    s = r[3] / ρ
    return m.μ / ρ * (1 - m.J2 * (m.R / ρ)^2 * (3s^2 - 1) / 2)
end

"""
    state_jacobian(model, x) -> SMatrix{6,6}

`A = ∂f/∂x` at `x`. It is analytic for `TwoBody` and for `CR3BP` (from
OrbitalMechanics), and comes from ForwardDiff for `J2Gravity`.
"""
state_jacobian(m::DynamicsModel, x::SVector{6}) = ForwardDiff.jacobian(y -> vector_field(m, y), x)

function state_jacobian(m::TwoBody, x::SVector{6,T}) where {T}
    # gravity gradient G = μ (3 r rᵀ / r⁵ - I / r³)
    r = SVector(x[1], x[2], x[3])
    r² = dot(r, r)
    r³ = r² * sqrt(r²)
    a = 3m.μ / (r² * r³)
    b = m.μ / r³
    # built entry by entry (column-major): block concatenation allocates here
    return SMatrix{6,6,T}(ntuple(Val(36)) do n
        i, j = mod1(n, 6), cld(n, 6)
        if i <= 3
            return i + 3 == j ? one(T) : zero(T)        # ∂ṙ/∂v = I
        elseif j <= 3
            return a * r[i-3] * r[j] - (i - 3 == j ? b : zero(T))   # ∂v̇/∂r = G
        else
            return zero(T)
        end
    end)
end

state_jacobian(m::CR3BP, x::SVector{6}) = cr3bp_jacobian(x, m.system.μ)

state_jacobian(m::DynamicsModel, x::AbstractVector) = state_jacobian(m, SVector{6}(x))

"""
    state_hessian(model, x) -> SArray{Tuple{6,6,6}}

`A2[i, l, m] = ∂²fᵢ/∂xₗ∂xₘ` at `x`, symmetric in `(l, m)`. It is analytic for
`TwoBody`. Other models use ForwardDiff on [`state_jacobian`](@ref), which for
`CR3BP` is the analytic Jacobian from OrbitalMechanics.
"""
function state_hessian(m::DynamicsModel, x::SVector{6})
    H = SArray{Tuple{6,6,6}}(ForwardDiff.jacobian(y -> vec(state_jacobian(m, y)), x))  # row = i + 6(l-1)
    # ∂A_il/∂x_m and ∂A_im/∂x_l come from different derivative paths, so they agree
    # only to rounding. Symmetrize: the exact Hessian is symmetric in (l, m).
    return SArray{Tuple{6,6,6}}(ntuple(n -> begin
        i, l, k = Tuple(CartesianIndices((6, 6, 6))[n])
        (H[i, l, k] + H[i, k, l]) / 2
    end, 216))
end

function state_hessian(m::TwoBody, x::SVector{6,T}) where {T}
    # ∂G_ij/∂r_k = μ [3(δ_ij r_k + δ_ik r_j + δ_jk r_i)/r⁵ - 15 r_i r_j r_k/r⁷]
    r = SVector(x[1], x[2], x[3])
    r² = dot(r, r)
    r⁵ = r² * r² * sqrt(r²)
    r⁷ = r⁵ * r²
    δ(a, b) = a == b ? one(T) : zero(T)
    H = zeros(MArray{Tuple{6,6,6},T})
    for i in 1:3, j in 1:3, k in 1:3
        H[3+i, j, k] = m.μ * (3 * (δ(i, j) * r[k] + δ(i, k) * r[j] + δ(j, k) * r[i]) / r⁵ -
                              15 * r[i] * r[j] * r[k] / r⁷)
    end
    return SArray(H)
end

state_hessian(m::DynamicsModel, x::AbstractVector) = state_hessian(m, SVector{6}(x))

"""
    symplectic_form(model) -> SMatrix{6,6}

The constant matrix `Ω` with `Φᵀ Ω Φ = Ω` for every state transition matrix `Φ`
of `model`, in the model's own `[r; v]` coordinates.

- `TwoBody`, `J2Gravity`: (r, v) are canonical (unit mass), so
  `Ω = J = [0 I; -I 0]`.
- `CR3BP`: the canonical momentum in the rotating frame is `p = v + ω × r` with
  `ω = ẑ`. So `[r; p] = T [r; v]`, with `T = [I 0; W I]` and `W r = ω × r`.
  Then `Ω = Tᵀ J T = [W - Wᵀ  I; -I  0]`. Using `J` here instead is wrong: the
  standard form is not preserved in (r, v).
"""
function symplectic_form(::Union{TwoBody,J2Gravity})
    Z = zero(SMatrix{3,3,Float64})
    E = SMatrix{3,3,Float64}(I)
    return [Z E; -E Z]
end

function symplectic_form(::CR3BP)
    W = @SMatrix [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]   # W r = ẑ × r
    Z = zero(SMatrix{3,3,Float64})
    E = SMatrix{3,3,Float64}(I)
    return [W-W' E; -E Z]
end

"""
    canonical_transform(model) -> SMatrix{6,6}

Matrix `T` that maps the model's `[r; v]` to canonical `[r; p]`. It is the
identity for the inertial models and `[I 0; W I]` for the `CR3BP`.
"""
canonical_transform(::Union{TwoBody,J2Gravity}) = SMatrix{6,6,Float64}(I)

function canonical_transform(::CR3BP)
    W = @SMatrix [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
    Z = zero(SMatrix{3,3,Float64})
    E = SMatrix{3,3,Float64}(I)
    return [E Z; W E]
end
