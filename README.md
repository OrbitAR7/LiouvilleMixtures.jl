# LiouvilleMixtures.jl

[![CI](https://github.com/OrbitAR7/LiouvilleMixtures.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/OrbitAR7/LiouvilleMixtures.jl/actions/workflows/CI.yml)

**Status: in development.** The dynamics, transition tensors, entropy/KL tools, the leading-order negentropy predictor ([derivations](docs/src/derivations.md)) and six propagation policies are implemented and tested. The study experiments are not yet run.

## Problems

Phase-space volume is conserved along the flows of the CR3BP, Kepler and J2 problems.
So the negentropy of a propagated Gaussian equals the log growth of its exact covariance
volume, and refitting a Gaussian can only add volume. This repository asks three questions:

- How large is that ratchet on the 9:2 NRHO?
- How much of the non-Gaussianity near perilune is reversible?
- Does a split and merge policy that responds only to non-Gaussianity that survives to the
  next measurement beat instantaneous triggers?

## Methods, Findings, Verification, Gaps

To be written as the study results come in.

## Reproduce

```bash
JULIA_NUM_THREADS=auto julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The suite takes about 2 minutes with 4 threads. Set `LIOUVILLE_FULL_TESTS=true` to add
the heavier statistical checks (CI runs them weekly). Monte Carlo propagation is
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
- [EstimationToolkit.jl](https://github.com/OrbitAR7/EstimationToolkit.jl), pinned to commit `9337396`. It supplies the
  three-component split library and Runnalls' merge bound.
- OrdinaryDiffEqVerner (the Vern9 integrator), ForwardDiff, Distributions (normal quantiles), StaticArrays.
- Test only: TaylorIntegration and TaylorSeries (jet transport, an independent check of the tensors).
