# =============================================================================
# State transition tensors by the variational equations
# =============================================================================
#
#   ẋ = f(x)
#   Φ̇ = A Φ                                       Φ(0) = I
#   Ψ̇_ijk = A_il Ψ_ljk + A2_ilm Φ_lj Φ_mk          Ψ(0) = 0
#
# with A = ∂f/∂x and A2 = ∂²f/∂x² evaluated along the trajectory. These are the
# first- and second-order state transition tensors of Park and Scheeres (JGCD
# 2006). Units follow the model: Φ_ij has units [x_i]/[x_j], and Ψ_ijk has units
# [x_i]/([x_j][x_k]).
#
# Integration uses Vern9 (Verner's 9(8) pair, OrdinaryDiffEqVerner) on a flat
# vector: 6 + 36 + 216 = 258 states at second order and 42 at first order.

const _N = 6
const _I_PHI = 7:42
const _I_PSI = 43:258

"""
    TransitionTensors

Result of [`propagate_tensors`](@ref). Fields:

- `t`: times, in the model's time unit.
- `x`: `Vector{SVector{6}}` of states.
- `Φ`: `Vector{Matrix}` of 6×6 state transition matrices.
- `Ψ`: `Vector{Array{Float64,3}}` of 6×6×6 second-order tensors. Empty if
  `order = 1`.
"""
struct TransitionTensors
    t::Vector{Float64}
    x::Vector{SVector{6,Float64}}
    Φ::Vector{Matrix{Float64}}
    Ψ::Vector{Array{Float64,3}}
end

function _rhs_order1!(du, u, model, t)
    x = SVector{6}(u[1], u[2], u[3], u[4], u[5], u[6])
    f = vector_field(model, x)
    A = state_jacobian(model, x)
    Φ = reshape(view(u, _I_PHI), _N, _N)
    dΦ = reshape(view(du, _I_PHI), _N, _N)
    @inbounds for i in 1:_N
        du[i] = f[i]
    end
    mul!(dΦ, A, Φ)
    return nothing
end

function _rhs_order2!(du, u, model, t)
    _rhs_order1!(du, u, model, t)
    x = SVector{6}(u[1], u[2], u[3], u[4], u[5], u[6])
    A = state_jacobian(model, x)
    A2 = state_hessian(model, x)
    Φ = SMatrix{6,6}(reshape(view(u, _I_PHI), _N, _N))
    Ψ = reshape(view(u, _I_PSI), _N, _N * _N)
    dΨ = reshape(view(du, _I_PSI), _N, _N * _N)
    mul!(dΨ, A, Ψ)                                   # A_il Ψ_ljk
    dΨ3 = reshape(dΨ, _N, _N, _N)
    @inbounds for i in 1:_N
        Hi = SMatrix{6,6}(A2[i, :, :])
        M = Φ' * Hi * Φ                              # (Φᵀ A2_i Φ)_jk
        # Exactly symmetric in (j, k); in floating point only to rounding, and on
        # the NRHO that rounding grows to ~2e-9 relative asymmetry of Ψ over one
        # revolution. Symmetrizing keeps Ψ_ijk = Ψ_ikj exactly.
        M = (M + M') / 2
        for k in 1:_N, j in 1:_N
            dΨ3[i, j, k] += M[j, k]
        end
    end
    return nothing
end

"""
    propagate_tensors(model, x0, t; order = 2, saveat = nothing,
                      reltol = 1e-13, abstol = 1e-13) -> TransitionTensors

Integrate the state, Φ and (for `order = 2`) Ψ from time 0 to `t`. The state and
`t` are in the units of `model`. `saveat` takes a vector of output times. By
default only the end point is returned.

Raises an error if the integrator does not reach `t`.
"""
function propagate_tensors(model::DynamicsModel, x0::AbstractVector, t::Real;
                           order::Int = 2, saveat = nothing,
                           reltol::Real = 1e-13, abstol::Real = 1e-13)
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2"))
    n = order == 1 ? last(_I_PHI) : last(_I_PSI)
    u0 = zeros(n)
    u0[1:6] .= x0
    for i in 1:_N
        u0[_N + i + _N * (i - 1)] = 1.0              # Φ(0) = I
    end
    rhs! = order == 1 ? _rhs_order1! : _rhs_order2!
    prob = ODEProblem(rhs!, u0, (0.0, float(t)), model)
    sol = saveat === nothing ?
        solve(prob, Vern9(); reltol, abstol, save_everystep = false, save_start = false) :
        solve(prob, Vern9(); reltol, abstol, saveat = collect(float.(saveat)))
    SciMLBase.successful_retcode(sol) ||
        error("propagate_tensors: integration failed with retcode $(sol.retcode)")
    xs = [SVector{6}(u[1:6]) for u in sol.u]
    Φs = [reshape(u[_I_PHI], _N, _N) for u in sol.u]
    Ψs = order == 1 ? Array{Float64,3}[] : [reshape(u[_I_PSI], _N, _N, _N) for u in sol.u]
    return TransitionTensors(collect(sol.t), xs, Φs, Ψs)
end

"""
    propagate_state(model, x0, t; reltol = 1e-13, abstol = 1e-13) -> SVector{6}

The flow `φ_t(x0)`, the state alone. Same integrator and units as
[`propagate_tensors`](@ref).
"""
function propagate_state(model::DynamicsModel, x0::AbstractVector, t::Real;
                         reltol::Real = 1e-13, abstol::Real = 1e-13)
    f(u, p, s) = vector_field(p, u)
    prob = ODEProblem{false}(f, SVector{6,Float64}(x0), (0.0, float(t)), model)
    sol = solve(prob, Vern9(); reltol, abstol, save_everystep = false, save_start = false)
    SciMLBase.successful_retcode(sol) ||
        error("propagate_state: integration failed with retcode $(sol.retcode)")
    return sol.u[end]
end

"""
    symplectic_residual(model, Φ; scales = nothing)

`maximum |Φᵀ Ω Φ - Ω|` with `Ω = symplectic_form(model)`. It is zero for an
exact Φ of a Hamiltonian flow.

Without `scales` the residual is in the model's own units. In km and km/s it
is unit-dependent: on a Molniya orbit |Φ| reaches 8×10⁵ s, and cancellation
in the velocity-velocity block leaves a floor near 5×10⁻⁹ whatever the
integration tolerance.

With `scales = (L, L, L, V, V, V)` the residual is computed in the
dimensionless state `x̃ = D⁻¹ x`, `D = diag(scales)`. There
`Φ̃ = D⁻¹ Φ D` preserves the same `Ω`, because a uniform length scale on r and
a uniform speed scale on v multiply the form by the constant L·V.
"""
function symplectic_residual(model::DynamicsModel, Φ::AbstractMatrix; scales = nothing)
    Ω = symplectic_form(model)
    if scales !== nothing
        D = Diagonal(collect(float.(scales)))
        Φ = D \ Φ * D
    end
    return maximum(abs, Φ' * Ω * Φ - Ω)
end
