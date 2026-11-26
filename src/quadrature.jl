# =============================================================================
# Gauss-Hermite quadrature for expectations under a Gaussian
# =============================================================================
#
# A deterministic reference for moments of φ(x), x ~ N(m, S Sᵀ), when the
# non-Gaussianity is too small for Monte Carlo to resolve (plan item 2c).

"""
    gauss_hermite(n) -> (nodes, weights)

The `n`-point Gauss-Hermite rule for the standard normal density
(probabilists' Hermite polynomials), with weights summing to 1. The rule is
exact for polynomials of degree up to `2n - 1`.

Computed by the Golub-Welsch algorithm (Math. Comp. 23, 1969). The nodes are the
eigenvalues of the symmetric tridiagonal Jacobi matrix with zero diagonal and
off-diagonal `√k`, k = 1 … n-1. The weights are the squared first components of
the normalized eigenvectors.
"""
function gauss_hermite(n::Integer)
    n >= 1 || throw(ArgumentError("n must be at least 1"))
    n == 1 && return [0.0], [1.0]
    J = SymTridiagonal(zeros(n), sqrt.(1.0:n-1))
    F = eigen(J)
    w = F.vectors[1, :] .^ 2
    return F.values, w / sum(w)
end

"""
    gauss_hermite_tensor(d, n) -> (Ξ, w)

Tensor-product rule in `d` dimensions for `N(0, I)`. `Ξ` is `d × nᵈ` (one node
per column) and `w` holds the weights, which sum to 1. It is exact for
polynomials of degree up to `2n - 1` in each variable. For d = 6 and n = 7 it
has 117 649 nodes.
"""
function gauss_hermite_tensor(d::Integer, n::Integer)
    x, w = gauss_hermite(n)
    N = n^d
    Ξ = Matrix{Float64}(undef, d, N)
    W = ones(N)
    for (j, I) in enumerate(CartesianIndices(ntuple(_ -> n, d)))
        for k in 1:d
            Ξ[k, j] = x[I[k]]
            W[j] *= w[I[k]]
        end
    end
    return Ξ, W
end

"""
    quadrature_mean_cov(Y, w) -> (μ, Σ)

Mean and covariance of `Y` (one image of a quadrature node per column) under
the quadrature weights `w`, which sum to 1. No `N - 1` correction is applied:
the rule computes expectations.
"""
function quadrature_mean_cov(Y::AbstractMatrix, w::AbstractVector)
    μ = Y * w
    D = Y .- μ
    return μ, Symmetric((D .* w') * D')
end
