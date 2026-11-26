# LiouvilleMixtures.jl

Entropy accounting for Gaussian and Gaussian-mixture uncertainty propagation
on volume-preserving orbital flows.

```@docs
LiouvilleMixtures
```

## Transition tensors

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

## Entropy tools

For a volume-preserving flow and a Gaussian prior, the negentropy of the
propagated density is

```math
J(t) = \mathrm{KL}\big(p_t \,\|\, \mathcal N(\mu_t, \Sigma_t)\big)
     = \tfrac12 \ln\det\Sigma_t - \tfrac12 \ln\det P_0 \qquad \text{(I2)},
```

so Monte Carlo needs only the sample covariance. [`negentropy_mc`](@ref)
estimates it with a bootstrap interval. By default it uses the sample covariance
of the same prior draws (the paired estimator). This cancels the sampling error of
the linear part, and on T0 it cuts the variance by a factor of 54 when J ≈ 0.008.

[`kl_mc`](@ref) scores any approximation `q` against the truth with the exact
prior log-density, `E[ln p₀(x₀) − ln |det Dφ| − ln q(φ(x₀))]`. It needs no density
estimation.

[`gauss_hermite`](@ref) builds the Golub-Welsch rule. Its tensor product is a
deterministic reference for `Σ_t` when the non-Gaussianity is too small for
Monte Carlo.
