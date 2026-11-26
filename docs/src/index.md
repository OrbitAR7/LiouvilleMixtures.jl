# LiouvilleMixtures.jl

Entropy accounting for Gaussian and Gaussian-mixture uncertainty propagation
on volume-preserving orbital flows.

```@docs
LiouvilleMixtures
```

## Transition tensors (milestone M1)

For `ẋ = f(x)` with `A = ∂f/∂x` and `A2 = ∂²f/∂x²`, the first- and second-order
state transition tensors obey

```math
\dot\Phi = A\,\Phi, \qquad
\dot\Psi_{ijk} = A_{il}\Psi_{ljk} + A2_{ilm}\,\Phi_{lj}\Phi_{mk},
\qquad \Phi(0) = I,\ \Psi(0) = 0.
```

[`propagate_tensors`](@ref) integrates them with Vern9. Each of the three models
(two-body, J2 and CR3BP) has `tr A = 0`, so `det Φ = 1` (Liouville).

**Symplectic form in the rotating frame.** The CR3BP state is the rotating-frame
`(r, v)`, and those coordinates are not canonical. The canonical momentum is
`p = v + ẑ × r`, so the preserved form is `Ω = Tᵀ J T = [W − Wᵀ  I; −I  0]`,
not `J` (see [`symplectic_form`](@ref)). On the 9:2 NRHO, `ΦᵀJΦ − J` is of
order 1 for an exact `Φ`.

**Checks** (test suite):

- Analytic Jacobians and Hessians are checked against ForwardDiff.
- J2 is checked against the gradient of its potential.
- `det Φ − 1 < 10⁻¹⁰` and the symplectic residual `< 10⁻⁹` are checked over one
  revolution of LEO, Molniya and the NRHO.
- Ψ is checked against central differences of Φ (`< 10⁻⁶`) and against jet
  transport with TaylorIntegration.jl (`< 10⁻⁸`).
