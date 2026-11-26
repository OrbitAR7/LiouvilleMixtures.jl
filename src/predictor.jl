# =============================================================================
# Leading-order negentropy from the whitened second-order transition tensor
# =============================================================================
#
# With x₀ = m₀ + S₀ ξ, ξ ~ N(0, I), and L = Φ S₀, the whitened deviation is
#
#     η = L⁻¹ (φ(x₀) - φ(m₀)) = ξ + ½ B(ξ, ξ) + O(|ξ|³),
#     B_abc = (L⁻¹)_ai Ψ_ijk (S₀)_jb (S₀)_kc.
#
# For a volume-preserving φ, the negentropy of the pushforward is, to leading
# order in the prior size,
#
#     J ≈ ¼ Σ B_abc² + ½ Σ B_abc B_bac = (1/12) ‖κ₃‖²,
#     κ₃_abc = B_abc + B_bca + B_cab        (standardized third cumulant).
#
# docs/derivations.md derives both forms. The first comes from the covariance
# (identity I2) together with the constraint that volume preservation places on
# the third-order tensor. The second comes from the Edgeworth expansion. The two
# agree identically for any B that is symmetric in its last two indices.
#
# Everything here is dimension-generic and dimensionless: B is built from
# whitened coordinates, so its units cancel.

"""
    whitened_tensor(Φ, Ψ, S0) -> Array{Float64,3}

The whitened second-order tensor

    B_abc = (L⁻¹)_ai Ψ_ijk (S0)_jb (S0)_kc,   L = Φ S0,

for a prior `N(m₀, S0 S0ᵀ)`.

- `Φ` and `Ψ` are the first- and second-order transition tensors at `m₀`, in
  any consistent units. `B` is dimensionless.
- `B` is symmetric in `(b, c)`.
- For a volume-preserving map, `Σ_a B_aac = 0` (see [`volume_trace`](@ref)).
"""
function whitened_tensor(Φ::AbstractMatrix, Ψ::AbstractArray{<:Real,3}, S0::AbstractMatrix)
    d = size(Φ, 1)
    size(Φ) == (d, d) && size(Ψ) == (d, d, d) && size(S0) == (d, d) ||
        throw(DimensionMismatch("Φ, Ψ and S0 must be d×d, d×d×d and d×d"))
    L = Φ * S0
    G = L \ reshape(Ψ, d, d * d)                # G[a, (j,k)] = (L⁻¹)_ai Ψ_ijk
    B = Array{Float64,3}(undef, d, d, d)
    for a in 1:d
        B[a, :, :] = S0' * reshape(G[a, :], d, d) * S0
    end
    return B
end

"""
    whitened_tensor(f, m0, S0) -> Array{Float64,3}

The whitened tensor of a map `y = f(x)` at `m0`, with `Φ` and `Ψ` taken from
ForwardDiff. `f` must accept and return vectors. It is used for the analytic
test maps (the T0 shear).
"""
function whitened_tensor(f, m0::AbstractVector, S0::AbstractMatrix)
    d = length(m0)
    Φ = ForwardDiff.jacobian(f, m0)
    Ψ = reshape(ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(f, y)), m0), d, d, d)
    return whitened_tensor(Φ, Ψ, S0)
end

"""
    negentropy_leading(B)

`¼ Σ B_abc² + ½ Σ B_abc B_bac`: the leading-order negentropy [nats] of
`η = ξ + ½ B(ξ, ξ)` under a volume-preserving map.
"""
function negentropy_leading(B::AbstractArray{<:Real,3})
    d = size(B, 1)
    s1 = sum(abs2, B)
    s2 = zero(eltype(B))
    for c in 1:d, b in 1:d, a in 1:d
        s2 += B[a, b, c] * B[b, a, c]
    end
    return s1 / 4 + s2 / 2
end

"""
    third_cumulant(B)

The standardized third cumulant of `η = ξ + ½ B(ξ, ξ)` to leading order:
`κ_abc = B_abc + B_bca + B_cab`. It is fully symmetric when `B` is symmetric in
its last two indices.
"""
third_cumulant(B::AbstractArray{<:Real,3}) = B .+ permutedims(B, (2, 3, 1)) .+ permutedims(B, (3, 1, 2))

"""
    negentropy_edgeworth(B)

`(1/12) ‖κ₃‖²` with `κ₃ = third_cumulant(B)`: the Edgeworth form of the
leading-order negentropy. It equals [`negentropy_leading`](@ref) for any `B`
that is symmetric in `(b, c)`.
"""
negentropy_edgeworth(B::AbstractArray{<:Real,3}) = sum(abs2, third_cumulant(B)) / 12

"""
    volume_trace(B) -> Vector

`Σ_a B_aac` for each `c`. It is zero for a volume-preserving map, because
`∂ ln det Φ / ∂x₀ = 0` gives `Σ_ij (Φ⁻¹)_ji Ψ_ijk = 0`. That is a check on Ψ
which does not depend on how Ψ was computed.
"""
volume_trace(B::AbstractArray{<:Real,3}) = [sum(B[a, a, c] for a in axes(B, 1)) for c in axes(B, 3)]

"""
    predicted_negentropy(model, m0, P0, t; S0 = cholesky(P0).L) -> Float64

Leading-order negentropy [nats] of `N(m0, P0)` propagated by `model` for time
`t`, from the second-order transition tensors at `m0`. The result does not
depend on which square root `S0` of `P0` is used.
"""
function predicted_negentropy(model::DynamicsModel, m0::AbstractVector, P0::AbstractMatrix, t::Real;
                              S0::AbstractMatrix = Matrix(cholesky(Symmetric(Matrix(P0))).L))
    sol = propagate_tensors(model, m0, t)
    return negentropy_leading(whitened_tensor(sol.Φ[end], sol.Ψ[end], S0))
end
