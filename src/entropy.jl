# =============================================================================
# Entropy, negentropy and KL divergence on volume-preserving flows
# =============================================================================
#
# All quantities are in nats. Samples are stored as the columns of a d × N
# matrix. Every function here is dimension-generic (d = 2 for the shear map T0,
# d = 6 for the orbit problems).
#
# Identities used (derived in docs/derivations.md, milestone M3):
#   For x_t = φ(x₀), x₀ ~ p₀, with a diffeomorphism φ:
#     H[p_t] = H[p₀] + E[ln |det Dφ(x₀)|]                     (change of variables)
#     ln p_t(φ(x₀)) = ln p₀(x₀) - ln |det Dφ(x₀)|              (I1 when det Dφ = 1)
#   Negentropy J = KL(p_t ‖ N(μ_t, Σ_t)) = ½ ln det(2πe Σ_t) - H[p_t].
#   For a Gaussian prior N(m₀, P₀) this is
#     J = ½ ln det Σ_t - ½ ln det P₀ - E[ln |det Dφ|]          (I2 when det Dφ = 1)

"""
    gaussian_entropy(Σ)

Differential entropy of `N(μ, Σ)` in nats: `½ ln det(2πe Σ)`.
"""
gaussian_entropy(Σ::AbstractMatrix) = (size(Σ, 1) * log(2π * ℯ) + logdet(Symmetric(Matrix(Σ)))) / 2

"""
    gaussian_logpdf(X, μ, Σ) -> Vector

Log-density of `N(μ, Σ)` at each column of `X` (one sample per column).
Uses a Cholesky factor of `Σ`. Raises an error if `Σ` is not positive definite.
"""
function gaussian_logpdf(X::AbstractMatrix, μ::AbstractVector, Σ::AbstractMatrix)
    C = cholesky(Symmetric(Matrix(Σ)))
    Z = C.L \ (X .- μ)
    d = size(X, 1)
    c = -(d * log(2π) + logdet(C)) / 2
    return [c - sum(abs2, view(Z, :, j)) / 2 for j in axes(Z, 2)]
end
gaussian_logpdf(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix) =
    gaussian_logpdf(reshape(x, :, 1), μ, Σ)[1]

"""
    sample_gaussian(rng, m, P, N) -> Matrix

`N` samples of `N(m, P)` as the columns of a `length(m) × N` matrix, drawn as
`m + L z` with `P = L Lᵀ` (Cholesky) and `z` standard normal.
"""
function sample_gaussian(rng::AbstractRNG, m::AbstractVector, P::AbstractMatrix, N::Integer)
    L = cholesky(Symmetric(Matrix(P))).L
    return m .+ L * randn(rng, length(m), N)
end

"""
    sample_mean_cov(X, w = nothing) -> (μ, Σ)

Mean and covariance of the columns of `X`. With integer bootstrap counts or
weights `w` (one per column), these are the weighted moments. The covariance
uses the unbiased `1/(N-1)` normalization for unit weights and
`1/(Σw - 1)` in general.
"""
function sample_mean_cov(X::AbstractMatrix, w = nothing)
    if w === nothing
        μ = vec(sum(X; dims = 2)) / size(X, 2)
        D = X .- μ
        return μ, Symmetric(D * D' / (size(X, 2) - 1))
    end
    # one pass, no d × N temporaries: this runs once per bootstrap replicate
    d, N = size(X)
    W = zero(Float64)
    μ = zeros(d)
    for j in 1:N
        wj = w[j]
        wj == 0 && continue
        W += wj
        @inbounds for k in 1:d
            μ[k] += wj * X[k, j]
        end
    end
    μ ./= W
    S = zeros(d, d)
    δ = zeros(d)
    for j in 1:N
        wj = w[j]
        wj == 0 && continue
        @inbounds for k in 1:d
            δ[k] = X[k, j] - μ[k]
        end
        @inbounds for l in 1:d, k in l:d
            S[k, l] += wj * δ[k] * δ[l]
        end
    end
    @inbounds for l in 1:d, k in 1:l-1
        S[k, l] = S[l, k]
    end
    return μ, Symmetric(S / (W - 1))
end

"""
    negentropy_from_covariance(Σt, P0; mean_logdetjac = 0)

Negentropy of `p_t = φ#N(m₀, P₀)` from its exact covariance `Σt`:

    J = ½ ln det Σt - ½ ln det P0 - E[ln |det Dφ|]

With the default `mean_logdetjac = 0` this is I2, which is exact for
volume-preserving `φ`. For a coordinate map `g` (equinoctial elements, M5),
pass the mean of `ln |det Dg|` over the samples.
"""
negentropy_from_covariance(Σt::AbstractMatrix, P0::AbstractMatrix; mean_logdetjac::Real = 0) =
    (logdet(Symmetric(Matrix(Σt))) - logdet(Symmetric(Matrix(P0)))) / 2 - mean_logdetjac

"""
    MCEstimate

A Monte Carlo estimate with a confidence interval. Fields:

- `value`: the estimate.
- `lo`, `hi`: the interval bounds.
- `se`: the standard error.
- `level`: the confidence level, for example 0.95.
- `n`: the sample size.
"""
struct MCEstimate
    value::Float64
    lo::Float64
    hi::Float64
    se::Float64
    level::Float64
    n::Int
end

Base.in(x::Real, e::MCEstimate) = e.lo <= x <= e.hi

"""
    bootstrap(stat, N; B = 1000, level = 0.95, rng) -> MCEstimate

Nonparametric bootstrap of a statistic of `N` iid samples.

- `stat(w)` must compute the statistic from integer resampling counts `w`
  (length `N`, summing to `N`). `stat(nothing)` must give the full-sample value.
- The interval is the percentile interval of `B` replicates. `se` is the
  standard deviation of the replicates.

Reference: Efron and Tibshirani, *An Introduction to the Bootstrap* (1993), ch. 13.
"""
function bootstrap(stat, N::Integer; B::Integer = 1000, level::Real = 0.95,
                   rng::AbstractRNG = Random.default_rng())
    θ = stat(nothing)
    reps = Vector{Float64}(undef, B)
    w = zeros(Int, N)
    for b in 1:B
        fill!(w, 0)
        for _ in 1:N
            w[rand(rng, 1:N)] += 1
        end
        reps[b] = stat(w)
    end
    sort!(reps)
    α = (1 - level) / 2
    lo = reps[clamp(round(Int, α * B), 1, B)]
    hi = reps[clamp(round(Int, (1 - α) * B), 1, B)]
    m = sum(reps) / B
    se = sqrt(sum(abs2, reps .- m) / (B - 1))
    return MCEstimate(θ, lo, hi, se, level, N)
end

"""
    mean_estimate(v; level = 0.95) -> MCEstimate

Sample mean of the iid terms `v`, with a normal-theory (CLT) confidence
interval: `mean ± z σ/√N`.
"""
function mean_estimate(v::AbstractVector; level::Real = 0.95)
    N = length(v)
    m = sum(v) / N
    se = sqrt(sum(abs2, v .- m) / (N - 1) / N)
    z = quantile(Normal(), (1 + level) / 2)
    return MCEstimate(m, m - z * se, m + z * se, se, level, N)
end

"""
    negentropy_mc(X0, Xt, P0; prior = :sample, logdetjac = nothing,
                  B = 1000, level = 0.95, rng) -> MCEstimate

Monte Carlo negentropy of the propagated samples `Xt = φ(X0)` by
[`negentropy_from_covariance`](@ref), with a bootstrap confidence interval over
the sample pairs.

- `prior = :exact` uses the known prior covariance `P0`.
- `prior = :sample` (the default) uses the sample covariance of the same `X0`.
  It is also consistent. Because Σ̂_t and Σ̂_0 share their sampling error, the
  linear part of the map cancels, and for small non-Gaussianity the variance is
  far lower.
- `logdetjac` takes per-sample `ln |det Dφ(x₀)|` values when `φ` is not exactly
  volume-preserving.
"""
function negentropy_mc(X0::AbstractMatrix, Xt::AbstractMatrix, P0::AbstractMatrix;
                       prior::Symbol = :sample, logdetjac = nothing,
                       B::Integer = 1000, level::Real = 0.95,
                       rng::AbstractRNG = Random.default_rng())
    size(X0) == size(Xt) || throw(DimensionMismatch("X0 and Xt must have the same size"))
    prior in (:sample, :exact) || throw(ArgumentError("prior must be :sample or :exact"))
    N = size(Xt, 2)
    function stat(w)
        _, Σt = sample_mean_cov(Xt, w)
        Σ0 = prior === :exact ? P0 : sample_mean_cov(X0, w)[2]
        ldj = logdetjac === nothing ? 0.0 :
              (w === nothing ? sum(logdetjac) / N : dot(w, logdetjac) / sum(w))
        return negentropy_from_covariance(Σt, Σ0; mean_logdetjac = ldj)
    end
    return bootstrap(stat, N; B, level, rng)
end

"""
    kl_mc(logp0, logq; logdetjac = nothing, level = 0.95) -> MCEstimate

Exact-log-density Monte Carlo estimate of `KL(p_t ‖ q)` for `p_t = φ#p₀`:

    KL = E over x₀ ~ p₀ of [ ln p₀(x₀) - ln |det Dφ(x₀)| - ln q(φ(x₀)) ]

- `logp0[j]` is `ln p₀` at sample j.
- `logq[j]` is `ln q` at the propagated sample j.
- `logdetjac` defaults to zero (volume-preserving φ, I1).

No density estimation is involved. The interval is normal-theory over the iid
terms.
"""
function kl_mc(logp0::AbstractVector, logq::AbstractVector; logdetjac = nothing,
               level::Real = 0.95)
    length(logp0) == length(logq) || throw(DimensionMismatch("logp0 and logq must have the same length"))
    terms = logdetjac === nothing ? logp0 .- logq : logp0 .- logdetjac .- logq
    return mean_estimate(terms; level)
end

"""
    propagate_samples(model, X0, t; logdet = false, reltol = 1e-13, abstol = 1e-13)

Propagate each column of the 6 × N matrix `X0` through the flow of `model` for
time `t`. Returns `Xt`. With `logdet = true` it returns `(Xt, ldj)`, where `ldj`
holds `ln |det Φ|` from each sample's own state transition matrix. Columns are
spread over `Threads.nthreads()` threads.
"""
function propagate_samples(model::DynamicsModel, X0::AbstractMatrix, t::Real;
                           logdet::Bool = false, reltol::Real = 1e-13, abstol::Real = 1e-13)
    size(X0, 1) == 6 || throw(DimensionMismatch("X0 must be 6 × N"))
    N = size(X0, 2)
    Xt = similar(X0, Float64)
    ldj = zeros(logdet ? N : 0)
    Threads.@threads for j in 1:N
        if logdet
            s = propagate_tensors(model, view(X0, :, j), t; order = 1, reltol, abstol)
            Xt[:, j] = s.x[end]
            ldj[j] = log(abs(det(s.Φ[end])))
        else
            Xt[:, j] = propagate_state(model, view(X0, :, j), t; reltol, abstol)
        end
    end
    return logdet ? (Xt, ldj) : Xt
end
