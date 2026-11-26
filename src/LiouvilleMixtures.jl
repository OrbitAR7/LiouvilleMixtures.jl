"""
    LiouvilleMixtures

Entropy accounting for Gaussian and Gaussian-mixture uncertainty propagation
on volume-preserving orbital flows.

Conventions used throughout the package:

- Kepler and J2 dynamics: inertial Cartesian state `[r; v]` in km and km/s,
  time in s.
- CR3BP dynamics: rotating-frame state `[r; v]` in the nondimensional units
  of `OrbitalMechanics.CR3BPSystem` (length `l_star`, time `t_star`). These
  coordinates are not canonical. The canonical momentum is `p = v + ω × r`.
- No function mixes dimensional and nondimensional quantities.
- Entropies and KL divergences are in nats.

The CR3BP equations of motion, the halo family and the 9:2 NRHO come from
OrbitalMechanics.jl (https://github.com/OrbitAR7/orbital-mechanics-jl). The
commit is pinned in `Project.toml`.
"""
module LiouvilleMixtures

using LinearAlgebra
using StaticArrays
using OrbitalMechanics

end # module
