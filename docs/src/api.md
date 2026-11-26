# API

## Dynamics models

```@docs
DynamicsModel
TwoBody
J2Gravity
CR3BP
vector_field
potential
state_jacobian
state_hessian
symplectic_form
canonical_transform
```

## Transition tensors

```@docs
TransitionTensors
propagate_tensors
propagate_state
symplectic_residual
```

## Test problems

```@docs
TestProblem
t1_leo
t1_molniya
nrho_9_2
t2_nrho
```

## Entropy, negentropy and KL divergence

```@docs
gaussian_entropy
gaussian_logpdf
sample_gaussian
sample_mean_cov
negentropy_from_covariance
negentropy_mc
kl_mc
MCEstimate
bootstrap
mean_estimate
propagate_samples
```

## Gauss-Hermite quadrature

```@docs
gauss_hermite
gauss_hermite_tensor
quadrature_mean_cov
```

## T0: shear map

```@docs
shear_map
shear_logpdf
t0_negentropy
t0_negentropy_leading
t0_kl_to_prior
```
