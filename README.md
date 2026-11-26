# LiouvilleMixtures.jl

[![CI](https://github.com/OrbitAR7/LiouvilleMixtures.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/OrbitAR7/LiouvilleMixtures.jl/actions/workflows/CI.yml)

**Status: in development. Milestones M1 (dynamics and transition tensors) and M2 (entropy and KL tools) are implemented; no study results yet.**

## Problems

Phase-space volume is conserved along the flows of the CR3BP, Kepler and J2 problems.
So the negentropy of a propagated Gaussian equals the log growth of its exact covariance
volume, and refitting a Gaussian can only add volume. This repository asks three questions:

- How large is that ratchet on the 9:2 NRHO?
- How much of the non-Gaussianity near perilune is reversible?
- Does a split and merge policy that responds only to non-Gaussianity that survives to the
  next measurement beat instantaneous triggers?

## Methods, Findings, Verification, Gaps

Filled in as milestones land.

## Reproduce

```bash
JULIA_NUM_THREADS=auto julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The suite takes about 2 minutes with 4 threads. Monte Carlo propagation is
threaded, and the results do not depend on the thread count.

Julia 1.10 (LTS) or later. On 1.10, install the pinned git dependencies first:

```bash
julia --project=. .github/install_sources.jl Project.toml
```

Build the documentation:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Dependencies

- [OrbitalMechanics.jl](https://github.com/OrbitAR7/orbital-mechanics-jl), pinned to commit
  `98a43ae` in `Project.toml`. It supplies the CR3BP equations of motion and the 9:2 NRHO.
- OrdinaryDiffEqVerner (the Vern9 integrator), ForwardDiff, Distributions (normal quantiles), StaticArrays.
- Test only: TaylorIntegration and TaylorSeries (jet transport, an independent check of the tensors).
