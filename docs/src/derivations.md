# Derivations

## Notation

- State ``x \in \mathbb R^d``, autonomous dynamics ``\dot x = f(x)``, flow ``\varphi_t``.
- Prior ``p_0 = \mathcal N(m_0, P_0)`` with ``P_0 = S_0 S_0^\top``.
- ``\Phi = \partial\varphi_t/\partial x_0`` and
  ``\Psi_{ijk} = \partial^2\varphi_{t,i}/\partial x_{0,j}\partial x_{0,k}``, both at ``m_0``.
- ``p_t = \varphi_{t\#}p_0`` is the propagated density. ``\mu_t`` and ``\Sigma_t`` are its exact
  mean and covariance. ``H`` is differential entropy in nats.
- Negentropy is ``J(t) = \mathrm{KL}(p_t\,\|\,\mathcal N(\mu_t,\Sigma_t))``, the information lost
  by replacing ``p_t`` with the Gaussian that has the same first two moments.
- Indices repeat with summation.

## Volume preservation

``\frac{d}{dt}\ln\det\Phi = \operatorname{tr}(\Phi^{-1}\dot\Phi) = \operatorname{tr}A`` with
``A = \partial f/\partial x``. So ``\det\Phi = 1`` for all ``t`` whenever ``\operatorname{tr}A = 0``.

That holds for each of the three models:
- **Two-body and J2 in inertial Cartesian ``(r, v)``.** ``A = \begin{bmatrix}0 & I\\ G & 0\end{bmatrix}``.
- **CR3BP in rotating ``(r, v)``.** The velocity block is the Coriolis matrix
  ``\begin{bmatrix}0&2&0\\-2&0&0\\0&0&0\end{bmatrix}``, which has zero trace.

Volume preservation does not make the rotating ``(r, v)`` canonical. The canonical momentum is
``p = v + \hat z\times r``. The form the flow preserves in ``(r, v)`` is
``\Omega = T^\top J T`` (see [`symplectic_form`](@ref)), not ``J``.

## I1: the density is carried along trajectories

For a diffeomorphism ``\varphi``, the change of variables gives

```math
p_t(\varphi(x_0)) = \frac{p_0(x_0)}{|\det D\varphi(x_0)|}, \qquad
H[p_t] = H[p_0] + \mathbb E_{p_0}\big[\ln|\det D\varphi(x_0)|\big].
```

With ``\det D\varphi = 1`` these become ``p_t(\varphi(x_0)) = p_0(x_0)`` and
``H[p_t] = H[p_0]``.

This is what makes the KL estimator in [`kl_mc`](@ref) exact. For any approximation ``q``,

```math
\mathrm{KL}(p_t\,\|\,q) = \mathbb E_{x_0\sim p_0}\big[\ln p_0(x_0) - \ln q(\varphi(x_0))\big].
```

It needs samples and two log-densities, and no density estimation.

## I2: negentropy is covariance volume

The Gaussian with covariance ``\Sigma`` has entropy ``\tfrac12\ln\det(2\pi e\Sigma)``. Because the
moment-matched Gaussian shares ``\mu_t`` and ``\Sigma_t`` with ``p_t``,
``\mathbb E_{p_t}[\ln\mathcal N(x;\mu_t,\Sigma_t)] = -\tfrac12\ln\det(2\pi e\Sigma_t)``. Therefore

```math
J(t) = -H[p_t] + \tfrac12\ln\det(2\pi e\Sigma_t)
     = \tfrac12\ln\det\Sigma_t - \tfrac12\ln\det P_0 - \mathbb E\ln|\det D\varphi|.
```

For a volume-preserving flow the last term vanishes. That leaves
``J = \tfrac12\ln\det\Sigma_t - \tfrac12\ln\det P_0``, so a Monte Carlo estimate of ``J`` needs
only a sample covariance.

The same formula with ``\varphi`` replaced by a coordinate map ``g`` gives the negentropy in
``g``-coordinates. There the Jacobian term does not vanish.

## I3: moment matching through a volume-preserving map cannot shrink the covariance volume

The Gaussian has the largest entropy of all densities with a given covariance. So
``H[p_t] \le \tfrac12\ln\det(2\pi e\Sigma_t)``, with equality only when ``p_t`` is Gaussian.
With ``H[p_t] = H[p_0]`` and a Gaussian input:

```math
\det\Sigma_t \;\ge\; \det P_0, \qquad \text{equality iff } p_t \text{ is Gaussian}.
```

This is the ratchet. Suppose a filter replaces the propagated density by its moment-matched
Gaussian at times ``t_1 < t_2 < \dots``. Each interval then starts from a Gaussian and ends with
a moment match, so ``\det\Sigma_{t_{k+1}} \ge \det\Sigma_{t_k}``. Its covariance volume cannot
decrease from one refit to the next.

The exact pushforward obeys only ``\det\Sigma_t \ge \det P_0``. Its volume can fall again as the
density unbends.

An unscented refit is not exact moment matching, so it can break the inequality by the size of
its moment error.

## I4: leading-order negentropy from the whitened second-order tensor

**Setup.** Write ``x_0 = m_0 + S_0\xi`` with ``\xi\sim\mathcal N(0, I)``, and whiten the output
with ``L = \Phi S_0``:

```math
\eta = L^{-1}\big(\varphi(x_0) - \varphi(m_0)\big)
     = \xi + \tfrac12 B(\xi,\xi) + \tfrac16 C(\xi,\xi,\xi) + O(\varepsilon^3),
\qquad
B_{abc} = (L^{-1})_{ai}\Psi_{ijk}(S_0)_{jb}(S_0)_{kc}.
```

Here ``\varepsilon`` is the prior size measured against the scale of the map's nonlinearity, so
``B = O(\varepsilon)`` and ``C = O(\varepsilon^2)``.

- ``B`` is symmetric in its last two indices, and ``C`` in its last three.
- ``\det D\eta = \det D\varphi/\det\Phi = 1`` exactly, so ``\eta`` is also a volume-preserving
  map of ``\xi``.
- By I2 in whitened coordinates, ``J = \tfrac12\ln\det\operatorname{Cov}(\eta)``.

**Claim.**

```math
J = \tfrac14 B_{abc}B_{abc} + \tfrac12 B_{abc}B_{bac} + O(\varepsilon^4)
  = \tfrac1{12}\|\kappa_3\|^2 + O(\varepsilon^4),
\qquad \kappa_{abc} = B_{abc} + B_{bca} + B_{cab}.
```

So the leading-order negentropy depends on ``B`` alone. The third-order tensor ``C`` drops out,
but only because volume preservation fixes the part of ``C`` that matters.

### Route 1: through the covariance (needs volume preservation)

**Moments of ξ.** Odd moments vanish, and
``\mathbb E[\xi_a\xi_b\xi_c\xi_d] = \delta_{ab}\delta_{cd} + \delta_{ac}\delta_{bd} + \delta_{ad}\delta_{bc}``.

**Covariance of η.** To ``O(\varepsilon^2)``,

```math
\operatorname{Cov}(\eta)_{ab}
 = \delta_{ab}
 + \tfrac14\operatorname{Cov}\!\big(B_a(\xi,\xi), B_b(\xi,\xi)\big)
 + \tfrac16\mathbb E\big[\xi_a C_b(\xi,\xi,\xi)\big] + \tfrac16\mathbb E\big[\xi_b C_a(\xi,\xi,\xi)\big]
 = \delta_{ab} + \tfrac12 B_{acd}B_{bcd} + \tfrac12 C_{baee} + \tfrac12 C_{abee}.
```

The cross terms between ``\xi`` and ``B(\xi,\xi)`` vanish because they are odd.

**Log-determinant.** With ``\tfrac12\ln\det(I + E) = \tfrac12\operatorname{tr}E + O(\varepsilon^4)``:

```math
J = \tfrac14 B_{acd}B_{acd} + \tfrac12 C_{aaee} + O(\varepsilon^4).
```

**Volume constraint.** Now use ``\det D\eta = 1``. Write ``D\eta = I + M`` with

```math
M_{ac} = B_{acd}\xi_d + \tfrac12 C_{acde}\xi_d\xi_e + \dots
```

Then ``\ln\det(I+M) = \operatorname{tr}M - \tfrac12\operatorname{tr}M^2 + \dots`` must vanish for
every ``\xi``:

- At ``O(\varepsilon)``: ``B_{aad} = 0`` for every ``d``. [`volume_trace`](@ref) tests this.
- At ``O(\varepsilon^2)``: ``\tfrac12 C_{aade}\xi_d\xi_e = \tfrac12 B_{acd}B_{cae}\xi_d\xi_e`` for
  every ``\xi``. Since ``B_{acd}B_{cae}`` is symmetric in ``(d, e)``, it follows that
  ``C_{aaee} = B_{ace}B_{cae}``.

**Result.** Substituting the constraint into the log-determinant:

```math
J = \tfrac14 B_{abc}B_{abc} + \tfrac12 B_{abc}B_{bac} + O(\varepsilon^4).
```

### Route 2: Edgeworth expansion (needs no volume preservation)

**Standardize.** ``\operatorname{Cov}(\eta) = I + O(\varepsilon^2)``. Standardizing therefore
changes the third cumulant only at ``O(\varepsilon^3)``.

**Third cumulant.** Let ``\tilde\eta = \eta - \mathbb E\eta``. Using the moments of ``\xi``,

```math
\kappa_{abc} = \mathbb E[\tilde\eta_a\tilde\eta_b\tilde\eta_c]
             = B_{abc} + B_{bca} + B_{cab} + O(\varepsilon^3).
```

**Expand the density.** Write the Gram-Charlier/Edgeworth form
``p(\eta) = \phi(\eta)\,(1 + g(\eta))`` with

```math
g = \tfrac16\kappa_{abc}H_{abc} + O(\varepsilon^2),
```

where ``H_{abc}`` are multivariate Hermite polynomials. Then
``\mathrm{KL} = \tfrac12\mathbb E_\phi[g^2] + O(\varepsilon^4)``:

- The ``O(\varepsilon^3)`` term ``\mathbb E_\phi[g^3]`` is the expectation of an odd polynomial,
  so it vanishes.
- The fourth-cumulant part of ``g`` is orthogonal to ``H_{abc}``.

**Result.** With ``\mathbb E_\phi[H_{abc}H_{def}]`` equal to the sum over the six pairings of the
two index triples, and ``\kappa`` fully symmetric,

```math
J = \tfrac1{72}\kappa_{abc}\kappa_{def}\,\mathbb E_\phi[H_{abc}H_{def}] = \tfrac1{12}\|\kappa_3\|^2 + O(\varepsilon^4).
```

### The two routes agree

Expand ``\|\kappa\|^2`` and use ``B_{bca} = B_{bac}`` (symmetry in the last two indices):

```math
\|\kappa\|^2 = 3B_{abc}B_{abc} + 6B_{abc}B_{bca} = 3B_{abc}B_{abc} + 6B_{abc}B_{bac},
```

so ``\tfrac1{12}\|\kappa\|^2 = \tfrac14 B_{abc}B_{abc} + \tfrac12 B_{abc}B_{bac}``. This holds for
every ``B`` that is symmetric in its last two indices.

The two routes therefore check each other:
- **Route 2** gives the negentropy of any smooth map to leading order.
- **Route 1** shows that for a volume-preserving map, the covariance growth that I2 equates with
  negentropy carries that same quantity, through the constraint on ``C``.
- **For a map that does not preserve volume,** route 2 still holds. But
  ``\tfrac12\ln\det\Sigma_t - \tfrac12\ln\det P_0`` then differs from ``J`` by
  ``\mathbb E\ln|\det D\varphi|``.

### Order of the error

Both routes leave an ``O(\varepsilon^4)`` remainder, so the relative error of the predictor is
``O(\varepsilon^2)``. "Leading order" means a small prior on a fixed map.

Weakening the map's nonlinearity while keeping a unit prior is a different limit. There the
higher Taylor coefficients of the map enter at the same order as ``B``, and the predictor is not
asymptotically exact. The test suite has a 2D example of this: the ratio tends to ``3/7``.

### Checks in the test suite

1. **T0 shear map.** ``B_{211} = 2c\sigma_1^2/\sigma_2`` gives ``J \approx c^2\sigma_1^4/\sigma_2^2``.
   That is the leading term of the exact ``\tfrac12\ln(1+2c^2\sigma_1^4/\sigma_2^2)``.
2. **A volume-preserving 2D map with ``B_{abc}B_{bac}\neq 0``.** The map is
   ``\eta_1 = x_1 + k(1-\cos x_1)``, ``\eta_2 = x_2/(1 + k\sin x_1)``, which T0 cannot test.
   - Its exact negentropy (by quadrature) divided by the prediction tends to 1, with an error
     proportional to ``s^2`` for prior ``s^2 I``.
   - Using only ``\tfrac14\sum B^2`` would be off by a factor of ``7/3``.
3. **Orbit problems (T1, T2).** The prediction is compared with the deterministic
   Gauss-Hermite and the Monte Carlo negentropy across a prior sweep. The sweep shows where the
   leading order stops being adequate.
