# Shared helpers for the test suite: independent routes to Φ and Ψ.
using TaylorIntegration, TaylorSeries

"Ψ in dimensionless form, Ψ̃_ijk = Ψ_ijk D_j D_k / D_i, with D the problem scales."
scaled(Ψ::AbstractArray{<:Real,3}, D) = [Ψ[i, j, k] * D[j] * D[k] / D[i] for i in 1:6, j in 1:6, k in 1:6]
scaled(Φ::AbstractMatrix, D) = [Φ[i, j] * D[j] / D[i] for i in 1:6, j in 1:6]
relerr(a, b) = norm(a - b) / norm(b)

"""
Ψ by central differences of the variational Φ: (Φ(x0 + h e_k) - Φ(x0 - h e_k)) / 2h,
with h_k = δ D_k.
"""
function fd_psi(p::TestProblem, t, δ)
    Ψ = zeros(6, 6, 6)
    for k in 1:6
        h = δ * p.scales[k]
        e = SVector{6}(ntuple(i -> i == k ? h : 0.0, 6))
        Φp = propagate_tensors(p.model, p.x0 + e, t; order = 1).Φ[end]
        Φm = propagate_tensors(p.model, p.x0 - e, t; order = 1).Φ[end]
        Ψ[:, :, k] = (Φp - Φm) / (2h)
    end
    return Ψ
end

"""
Φ and Ψ by jet transport: integrate x0 + δx with δx a second-order TaylorN in six
variables (TaylorIntegration.jl), then read off the first and second derivatives
at δx = 0. The variables are scaled, x0 + D ⊙ ξ, to keep the polynomial
coefficients of similar size. The derivatives are unscaled on return.
"""
function jet_tensors(p::TestProblem, t; order = 28, abstol = 1e-22)
    ξ = variables!("ξ"; numvars = 6, order = 2, nowarn = true)
    q0 = [p.x0[i] + p.scales[i] * ξ[i] for i in 1:6]
    function f!(dq, q, model, s)
        v = vector_field(model, SVector{6}(q[1], q[2], q[3], q[4], q[5], q[6]))
        for i in 1:6
            dq[i] = v[i]
        end
        return nothing
    end
    sol = taylorinteg(f!, q0, 0.0, t, order, abstol, p.model; maxsteps = 200_000)
    xf = sol.x[end, :]
    D = p.scales
    Φ = [constant_term(differentiate(xf[i], j)) / D[j] for i in 1:6, j in 1:6]
    Ψ = [constant_term(differentiate(differentiate(xf[i], j), k)) / (D[j] * D[k])
         for i in 1:6, j in 1:6, k in 1:6]
    return Φ, Ψ, sol.t[end]
end
