# =============================================================================
# Gaussian mixtures: moments, log-density, split, merge
# =============================================================================
#
# States are 6-vectors in the units of the dynamics model. A mixture's weights
# always sum to 1. Splits use the three-component moment-matched library of
# EstimationToolkit.jl (`split_gaussian`). Merges replace two components by
# their moment-matched Gaussian, and their cost is Runnalls' bound
# (EstimationToolkit `merge_cost`).

"""
    GaussianMixture(w, m, P)
    GaussianMixture(m, P)

Mixture `Σᵢ wᵢ N(mᵢ, Pᵢ)` with 6-dimensional components.

- The weights are normalized on construction. An error is raised if any weight
  is negative or the sum is not positive.
- `GaussianMixture(m, P)` is the single Gaussian.
"""
struct GaussianMixture
    w::Vector{Float64}
    m::Vector{SVector{6,Float64}}
    P::Vector{SMatrix{6,6,Float64,36}}
    function GaussianMixture(w::AbstractVector, m::AbstractVector, P::AbstractVector)
        length(w) == length(m) == length(P) || throw(DimensionMismatch("w, m and P must have the same length"))
        all(>=(0), w) && sum(w) > 0 || throw(ArgumentError("weights must be non-negative with a positive sum"))
        new(collect(float.(w)) ./ sum(w), [SVector{6,Float64}(x) for x in m],
            [SMatrix{6,6,Float64,36}((Σ + Σ') / 2) for Σ in P])
    end
end
GaussianMixture(m::AbstractVector{<:Real}, P::AbstractMatrix) = GaussianMixture([1.0], [m], [P])

"""
    ncomponents(gm)

Number of components of the mixture `gm`.
"""
ncomponents(gm::GaussianMixture) = length(gm.w)

"""
    mixture_mean_cov(gm) -> (μ, Σ)

Mean and covariance of the mixture:
`μ = Σ wᵢ mᵢ` and `Σ = Σ wᵢ (Pᵢ + (mᵢ - μ)(mᵢ - μ)ᵀ)`.
"""
function mixture_mean_cov(gm::GaussianMixture)
    μ = sum(gm.w[i] * gm.m[i] for i in eachindex(gm.w))
    Σ = sum(gm.w[i] * (gm.P[i] + (gm.m[i] - μ) * (gm.m[i] - μ)') for i in eachindex(gm.w))
    return μ, (Σ + Σ') / 2
end

"""
    mixture_logpdf(gm, X) -> Vector

Log-density of the mixture at each column of `X`. The component log-densities are
combined with log-sum-exp, which avoids underflow far from all components.
"""
function mixture_logpdf(gm::GaussianMixture, X::AbstractMatrix)
    L = reduce(hcat, [log(gm.w[i]) .+ gaussian_logpdf(X, gm.m[i], gm.P[i]) for i in eachindex(gm.w)])
    return [_logsumexp(view(L, j, :)) for j in axes(L, 1)]
end

function _logsumexp(v)
    a = maximum(v)
    isfinite(a) || return a
    return a + log(sum(exp.(v .- a)))
end

"""
    split_component(gm, i, u; shrink = 0.7) -> GaussianMixture

Replace component `i` by three components along the direction `S u`.

- `S` is the Cholesky factor of `Pᵢ`, and `u` is a direction in the component's
  whitened coordinates (normalized here).
- The split is `EstimationToolkit.split_gaussian`: weights 1/6, 2/3, 1/6, offsets
  ±√(3(1 - shrink²)) σ, and covariance `Pᵢ - (1 - shrink²) s sᵀ`.
- It preserves the component's mean and covariance, and so the mixture's.
"""
function split_component(gm::GaussianMixture, i::Integer, u::AbstractVector; shrink::Real = 0.7)
    S = cholesky(Symmetric(Matrix(gm.P[i]))).L
    s = SVector{6,Float64}(S * (u / norm(u)))
    means, Pc, wts = split_gaussian(gm.m[i], gm.P[i], s; shrink)
    w = vcat(gm.w[1:i-1], gm.w[i] .* collect(wts), gm.w[i+1:end])
    m = vcat(gm.m[1:i-1], collect(means), gm.m[i+1:end])
    P = vcat(gm.P[1:i-1], fill(SMatrix{6,6,Float64,36}(Pc), 3), gm.P[i+1:end])
    return GaussianMixture(w, m, P)
end

"""
    merge_components(gm, i, j) -> GaussianMixture

Replace components `i` and `j` by their moment-matched Gaussian. The mixture's
mean and covariance are unchanged.
"""
function merge_components(gm::GaussianMixture, i::Integer, j::Integer)
    i == j && throw(ArgumentError("cannot merge a component with itself"))
    w = gm.w[i] + gm.w[j]
    μ = (gm.w[i] * gm.m[i] + gm.w[j] * gm.m[j]) / w
    di, dj = gm.m[i] - μ, gm.m[j] - μ
    P = (gm.w[i] * (gm.P[i] + di * di') + gm.w[j] * (gm.P[j] + dj * dj')) / w
    keep = [k for k in eachindex(gm.w) if k != i && k != j]
    return GaussianMixture(vcat(gm.w[keep], w), vcat(gm.m[keep], [μ]), vcat(gm.P[keep], [P]))
end

"""
    merge_cost(gm, i, j)

Runnalls' (2007) upper bound on the KL divergence added by merging components
`i` and `j`: `½[(wᵢ + wⱼ) ln det P_ij - wᵢ ln det Pᵢ - wⱼ ln det Pⱼ]`, where
`P_ij` is the merged covariance. It is computed by
`EstimationToolkit.merge_cost`.
"""
merge_cost(gm::GaussianMixture, i::Integer, j::Integer) =
    EstimationToolkit.merge_cost(gm.w[i], gm.m[i], gm.P[i], gm.w[j], gm.m[j], gm.P[j])
