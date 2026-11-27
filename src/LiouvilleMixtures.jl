"""
    LiouvilleMixtures

Entropy accounting for Gaussian and Gaussian-mixture uncertainty propagation
on volume-preserving orbital flows.

Conventions used throughout the package:

- Kepler and J2 dynamics: inertial Cartesian state `[r; v]` in km and km/s,
  time in s.
- CR3BP dynamics: rotating-frame state `[r; v]` in the nondimensional units
  of `OrbitalMechanics.CR3BPSystem` (length `l_star`, time `t_star`). These
  coordinates are not canonical. The canonical momentum is `p = v + ω × r`
  (see [`symplectic_form`](@ref)).
- No function mixes dimensional and nondimensional quantities.
- Entropies and KL divergences are in nats.

The CR3BP equations of motion, the halo family, the 9:2 NRHO and the physical
constants come from OrbitalMechanics.jl
(https://github.com/OrbitAR7/orbital-mechanics-jl). The commit is pinned in
`Project.toml`.
"""
module LiouvilleMixtures

using LinearAlgebra
using Random
using Distributions: Normal, quantile
using StaticArrays
using ForwardDiff
using OrdinaryDiffEqVerner
using OrbitalMechanics
import EstimationToolkit
using EstimationToolkit: split_gaussian
using PrecompileTools: @setup_workload, @compile_workload

export DynamicsModel, TwoBody, J2Gravity, CR3BP
export vector_field, potential, state_jacobian, state_hessian
export symplectic_form, canonical_transform
export TransitionTensors, propagate_tensors, propagate_state, symplectic_residual, symplectic_inverse
export TestProblem, t1_leo, t1_molniya, t2_nrho, nrho_9_2
export shear_map, shear_logpdf, t0_negentropy, t0_negentropy_leading, t0_kl_to_prior
export gaussian_entropy, gaussian_logpdf, sample_gaussian, sample_mean_cov
export negentropy_from_covariance, MCEstimate, bootstrap, mean_estimate
export negentropy_mc, kl_mc, propagate_samples
export gauss_hermite, gauss_hermite_tensor, quadrature_mean_cov
export whitened_tensor, negentropy_leading, third_cumulant, negentropy_edgeworth
export volume_trace, predicted_negentropy
export measurement_tensors, measurement_negentropy, max_nonlinearity_direction
export GaussianMixture, ncomponents, mixture_mean_cov, mixture_logpdf
export split_component, merge_components, merge_cost
export MeasurementModel, RangeRangeRate, Angles, measure, measurement_cov
export measurement_jacobian, measurement_hessian
export Policy, LinearPropagation, ScheduledRefit, EntropyTrigger, HorizonSplit, propagate_policy

include("dynamics.jl")
include("tensors.jl")
include("test_problems.jl")
include("entropy.jl")
include("quadrature.jl")
include("measurements.jl")
include("predictor.jl")
include("mixtures.jl")
include("policies.jl")
include("precompile.jl")

end # module
