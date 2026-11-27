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

# -----------------------------------------------------------------------------
# Measurement-space criterion and direction of largest nonlinearity
# -----------------------------------------------------------------------------

"""
    measurement_tensors(meas, Φ, Ψ, x̂, S0) -> (A, Q, Pz)

Whitened first- and second-order terms of the composed map
`ξ ↦ z = h(φ(m₀ + S0 ξ))` from the whitened prior to the measurement.

- `x̂ = φ(m₀)`, where `Φ` and `Ψ` are the transition tensors of `φ` at `m₀`.
- `Pz = G₁ S0 S0ᵀ G₁ᵀ + R`, with `G₁ = H Φ`, is the predicted measurement
  covariance (linear).
- With `W = chol(Pz).L⁻¹`, the outputs are `A = W G₁ S0` and
  `Q[c, :, :] = S0ᵀ (W G₂)_c S0`, where
  `G₂_ijk = H_il Ψ_ljk + H2_ilm Φ_lj Φ_mk`.
"""
function measurement_tensors(meas::MeasurementModel, Φ::AbstractMatrix, Ψ::AbstractArray{<:Real,3},
                             x̂::AbstractVector, S0::AbstractMatrix)
    H = measurement_jacobian(meas, x̂)
    H2 = measurement_hessian(meas, x̂)
    nz = size(H, 1)
    G1 = H * Φ
    G2 = reshape(H * reshape(Ψ, 6, 36), nz, 6, 6)
    for i in 1:nz
        G2[i, :, :] .+= Φ' * H2[i, :, :] * Φ
    end
    Pz = G1 * S0 * S0' * G1' + Matrix(measurement_cov(meas))
    W = inv(cholesky(Symmetric(Pz)).L)
    A = W * G1 * S0
    WG2 = reshape(W * reshape(G2, nz, 36), nz, 6, 6)
    Q = similar(WG2)
    for c in 1:nz
        Q[c, :, :] = S0' * WG2[c, :, :] * S0
    end
    return A, Q, Pz
end

"""
    measurement_negentropy(A, Q)

Leading-order negentropy [nats] of the predicted measurement, `(1/12) ‖κ₃‖²`, with

    κ_abc = (A Q_c Aᵀ)_ab + (A Q_a Aᵀ)_bc + (A Q_b Aᵀ)_ca,

the standardized third cumulant of `z` to leading order. The measurement noise is
Gaussian, so it adds no third cumulant; it enters only through the whitening in
`A` and `Q`. With `h` the identity and `R → 0`, this reduces to
[`negentropy_leading`](@ref) of the state.
"""
function measurement_negentropy(A::AbstractMatrix, Q::AbstractArray{<:Real,3})
    nz = size(A, 1)
    M = [A * Q[c, :, :] * A' for c in 1:nz]             # M[c][a, b] = (A Q_c Aᵀ)_ab
    κ = [M[c][a, b] + M[a][b, c] + M[b][c, a] for a in 1:nz, b in 1:nz, c in 1:nz]
    return sum(abs2, κ) / 12
end

"""
    max_nonlinearity_direction(T; iters = 500, tol = 1e-12) -> (u, f)

Unit vector `u` that maximizes `f(u) = Σ_c (uᵀ T_c u)²` for the stack of
symmetric matrices `T[c, :, :]`: the input direction along which a quadratic map
bends most.

The search is a shifted higher-order power iteration,
`u ← normalize(Σ_c (uᵀT_c u) T_c u + β u)` with `β = 3 Σ_c ‖T_c‖²`. The fixed
point of the unshifted step is a stationary point of f. The shift makes the
iteration monotone in f, as in SS-HOPM for symmetric tensors. It starts from
every coordinate axis and from the leading eigenvector of `Σ_c T_c²`, and keeps
the best local maximum.
"""
function max_nonlinearity_direction(T::AbstractArray{<:Real,3}; iters::Integer = 500, tol::Real = 1e-12)
    k, d = size(T, 1), size(T, 2)
    Ts = [T[c, :, :] for c in 1:k]
    f(u) = sum((dot(u, Tc * u))^2 for Tc in Ts)
    M = sum(Tc * Tc for Tc in Ts)
    starts = [collect(eachcol(Matrix(1.0I, d, d)))..., eigen(Symmetric(M)).vectors[:, end]]
    best_u, best_f = starts[1], -Inf
    β = 3 * sum(opnorm(Tc)^2 for Tc in Ts)
    β == 0 && return starts[1], 0.0
    for u0 in starts
        u = u0 / norm(u0)
        for _ in 1:iters
            g = sum(dot(u, Tc * u) * (Tc * u) for Tc in Ts) + β * u
            v = g / norm(g)
            done = norm(v - u) < tol
            u = v
            done && break
        end
        fu = f(u)
        fu > best_f && ((best_u, best_f) = (u, fu))
    end
    return best_u, best_f
end
