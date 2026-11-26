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
using StaticArrays
using ForwardDiff
using OrdinaryDiffEqVerner
using OrbitalMechanics

export DynamicsModel, TwoBody, J2Gravity, CR3BP
export vector_field, potential, state_jacobian, state_hessian
export symplectic_form, canonical_transform
export TransitionTensors, propagate_tensors, propagate_state, symplectic_residual
export TestProblem, t1_leo, t1_molniya, t2_nrho, nrho_9_2

include("dynamics.jl")
include("tensors.jl")
include("test_problems.jl")

end # module
