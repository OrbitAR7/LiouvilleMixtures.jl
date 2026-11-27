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
propagate_trajectory
symplectic_residual
symplectic_inverse
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
logdet_sample_cov
logdet_quadrature_cov
negentropy_from_covariance
negentropy_mc
kl_mc
MCEstimate
bootstrap
mean_estimate
propagate_samples
```

## Modified equinoctial elements

```@docs
cartesian_to_equinoctial
equinoctial_to_cartesian
equinoctial_logdetjac
unwrap_angle
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

## Negentropy predictor

```@docs
whitened_tensor
negentropy_leading
third_cumulant
negentropy_edgeworth
volume_trace
predicted_negentropy
measurement_tensors
measurement_negentropy
max_nonlinearity_direction
```

## Gaussian mixtures

```@docs
GaussianMixture
ncomponents
mixture_mean_cov
mixture_logpdf
split_component
merge_components
merge_cost
```

## Measurement models

```@docs
MeasurementModel
RangeRangeRate
Angles
measure
measurement_cov
measurement_jacobian
measurement_hessian
```

## Propagation policies

```@docs
Policy
LinearPropagation
ScheduledRefit
EntropyTrigger
HorizonSplit
propagate_policy
```
