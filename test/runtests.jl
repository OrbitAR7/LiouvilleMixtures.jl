using LiouvilleMixtures
using OrbitalMechanics
using LinearAlgebra
using StaticArrays
using ForwardDiff
using Random
using Test

include("helpers.jl")

# the 9:2 NRHO takes a few seconds to build, so build it once
# Full tier: heavier statistical checks that duplicate what the fast suite already
# covers. CI runs them weekly and on manual dispatch:
#   LIOUVILLE_FULL_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'
const FULL_TESTS = get(ENV, "LIOUVILLE_FULL_TESTS", "false") == "true"
@info "LiouvilleMixtures tests: $(FULL_TESTS ? "full" : "fast") tier, $(Threads.nthreads()) threads"

const NRHO = t2_nrho()
const NRHO_L = NRHO.model.system.l_star                 # km
const NRHO_V = NRHO.model.system.l_star / NRHO.model.system.t_star   # km/s

@testset "LiouvilleMixtures" verbose = true begin
    @testset "pinned OrbitalMechanics" begin
        # Earth-Moon mass ratio, about 0.01215 (GM_moon / (GM_earth + GM_moon))
        @test 0.0121 < EARTH_MOON.μ < 0.0122
    end
    include("test_dynamics.jl")
    include("test_tensors.jl")
    include("test_elements.jl")
    include("test_entropy.jl")
    include("test_predictor.jl")
    include("test_policies.jl")
end
