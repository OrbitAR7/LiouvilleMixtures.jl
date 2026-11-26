using LiouvilleMixtures
using OrbitalMechanics
using Test

@testset "LiouvilleMixtures" begin
    # M0 smoke tests: the package loads, and the pinned dependency provides
    # the CR3BP pieces that M1 builds on.
    @testset "package loads" begin
        @test isa(LiouvilleMixtures, Module)
    end

    @testset "pinned OrbitalMechanics" begin
        # Earth-Moon mass ratio, about 0.01215 (GM_moon / (GM_earth + GM_moon))
        @test 0.0121 < EARTH_MOON.μ < 0.0122
        @test isdefined(OrbitalMechanics, :cr3bp_eom)
        @test isdefined(OrbitalMechanics, :cr3bp_jacobian)
        @test isdefined(OrbitalMechanics, :orbit_with_period)
    end
end
