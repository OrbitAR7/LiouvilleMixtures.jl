using LiouvilleMixtures
using OrbitalMechanics
using LinearAlgebra
using StaticArrays
using ForwardDiff
using Random
using Test

include("helpers.jl")

# the 9:2 NRHO takes a few seconds to build, so build it once
const NRHO = t2_nrho()

@testset "LiouvilleMixtures" begin
    @testset "pinned OrbitalMechanics" begin
        # Earth-Moon mass ratio, about 0.01215 (GM_moon / (GM_earth + GM_moon))
        @test 0.0121 < EARTH_MOON.μ < 0.0122
    end
    include("test_dynamics.jl")
    include("test_tensors.jl")
end
