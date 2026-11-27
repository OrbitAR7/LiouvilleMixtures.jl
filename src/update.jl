# =============================================================================
# Measurement update of a Gaussian mixture, and the exact posterior by
# importance weighting
# =============================================================================
#
# z = h(x) + v, v ~ N(0, R). Units and frame are those of the dynamics model
# (see measurements.jl). The angle components of `Angles` are wrapped into
# (-π, π] wherever a difference of measurements is formed.

"""
    residual(meas, z, ẑ)

Measurement difference `z - ẑ`. For [`Angles`](@ref) the azimuth difference is
wrapped into (-π, π].
"""
residual(::MeasurementModel, z::AbstractVector, ẑ::AbstractVector) = SVector{length(z)}(z) - SVector{length(z)}(ẑ)
residual(::Angles, z::AbstractVector, ẑ::AbstractVector) =
    SVector(rem2pi(z[1] - ẑ[1], RoundNearest), z[2] - ẑ[2])

"""
    measurement_update(gm, meas, z) -> GaussianMixture

Gaussian-sum update of `gm` with the measurement `z` (Alspach and Sorenson, IEEE
TAC 1972). Each component gets an extended Kalman update at its own mean:

    H = ∂h/∂x(mᵢ),  S = H Pᵢ Hᵀ + R,  K = Pᵢ Hᵀ S⁻¹,
    mᵢ⁺ = mᵢ + K r,  Pᵢ⁺ = (I - K H) Pᵢ (I - K H)ᵀ + K R Kᵀ   (Joseph form),

with `r = residual(meas, z, h(mᵢ))`. Each weight is multiplied by the predicted
likelihood `N(r; 0, S)`. This is done in log space and then normalized, so a
measurement far from every component does not underflow.
"""
function measurement_update(gm::GaussianMixture, meas::MeasurementModel, z::AbstractVector)
    R = Matrix(measurement_cov(meas))
    n = ncomponents(gm)
    logw = zeros(n)
    m⁺ = similar(gm.m)
    P⁺ = similar(gm.P)
    for i in 1:n
        m, P = gm.m[i], Matrix(gm.P[i])
        H = Matrix(measurement_jacobian(meas, m))
        r = Vector(residual(meas, z, measure(meas, m)))
        S = Symmetric(H * P * H' + R)
        C = cholesky(S)
        K = (C \ (H * P))'
        A = I - K * H
        m⁺[i] = m + SVector{6}(K * r)
        P⁺[i] = SMatrix{6,6}(Symmetric(A * P * A' + K * R * K'))
        logw[i] = log(gm.w[i]) - (length(r) * log(2π) + logdet(C) + dot(r, C \ r)) / 2
    end
    w = exp.(logw .- maximum(logw))
    return GaussianMixture(w ./ sum(w), m⁺, P⁺)
end

"""
    measurement_loglik(meas, z, X) -> Vector

`ln p(z | x)` for each column of `X`: the Gaussian measurement log-likelihood
`ln N(residual(z, h(x)); 0, R)`.
"""
function measurement_loglik(meas::MeasurementModel, z::AbstractVector, X::AbstractMatrix)
    C = cholesky(Symmetric(Matrix(measurement_cov(meas))))
    nz = length(z)
    c = -(nz * log(2π) + logdet(C)) / 2
    return [begin
        r = Vector(residual(meas, z, measure(meas, view(X, :, j))))
        c - dot(r, C \ r) / 2
    end for j in axes(X, 2)]
end

"""
    posterior_kl(logp0, loglik, logq) -> (kl, ess)

Importance-sampling estimate of `KL(p(x | z) ‖ q)` from samples `x₀ ~ p₀` that
have been propagated to `x = φ(x₀)`:

- `logp0[j] = ln p₀(x₀ʲ)`. This is the exact prior log-density, and by I1 it
  equals `ln p_t(xʲ)` for a volume-preserving flow.
- `loglik[j] = ln p(z | xʲ)`.
- `logq[j]` is the approximate posterior log-density at `xʲ`.

The exact posterior is `p(x | z) = p_t(x) p(z | x) / p(z)`, with
`p(z) ≈ mean p(z | xʲ)`. Weighting by `w̄ⱼ ∝ p(z | xʲ)` gives

    KL ≈ Σⱼ w̄ⱼ [ln p₀(x₀ʲ) + ln p(z | xʲ) - ln p(z) - ln q(xʲ)].

`ess = 1 / Σ w̄ⱼ²` is the effective sample size. Results with a small ESS should
not be trusted.
"""
function posterior_kl(logp0::AbstractVector, loglik::AbstractVector, logq::AbstractVector)
    N = length(logp0)
    length(loglik) == N == length(logq) || throw(DimensionMismatch("inputs must have the same length"))
    lse = _logsumexp(loglik)
    logpz = lse - log(N)
    w = exp.(loglik .- lse)
    kl = sum(w .* (logp0 .+ loglik .- logpz .- logq))
    return kl, 1 / sum(abs2, w)
end
