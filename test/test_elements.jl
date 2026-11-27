# Modified equinoctial elements and multi-time propagation.

@testset "equinoctial elements and trajectories" begin
    μ = GM_EARTH
    @testset "against classical elements and round trip" begin
        for (a, e, i, Ω, ω, ν) in ((7000.0, 0.01, 51.6, 0.0, 0.0, 0.0), (26554.0, 0.72, 63.4, 40.0, 270.0, 100.0),
                                   (42164.0, 1e-4, 0.1, 75.0, 10.0, 250.0), (9000.0, 0.3, 120.0, 300.0, 45.0, 181.0))
            oe = KeplerianElements(a, e, deg2rad(i), deg2rad(Ω), deg2rad(ω), deg2rad(ν))
            s = keplerian_to_cartesian(oe; μ)
            x = SVector{6}(s.r..., s.v...)
            q = cartesian_to_equinoctial(x, μ)
            I, O, W = deg2rad(i), deg2rad(Ω), deg2rad(ω)
            @test q[1] ≈ a * (1 - e^2) rtol = 1e-12
            @test q[2] ≈ e * cos(W + O) atol = 1e-12
            @test q[3] ≈ e * sin(W + O) atol = 1e-12
            @test q[4] ≈ tan(I / 2) * cos(O) rtol = 1e-10 atol = 1e-12
            @test q[5] ≈ tan(I / 2) * sin(O) rtol = 1e-10 atol = 1e-12
            @test cos(q[6]) ≈ cos(O + W + deg2rad(ν)) atol = 1e-12
            @test sin(q[6]) ≈ sin(O + W + deg2rad(ν)) atol = 1e-12
            @test equinoctial_to_cartesian(q, μ) ≈ x rtol = 1e-12
        end
    end

    @testset "Jacobian log-determinant" begin
        x = t1_molniya().x0
        J = ForwardDiff.jacobian(y -> cartesian_to_equinoctial(y, GM_EARTH), x)
        Jinv = ForwardDiff.jacobian(q -> equinoctial_to_cartesian(q, GM_EARTH), cartesian_to_equinoctial(x, GM_EARTH))
        @test J * Jinv ≈ I atol = 1e-9
        @test equinoctial_logdetjac(x, GM_EARTH) ≈ -logabsdet(Jinv)[1] rtol = 1e-10
    end

    @testset "angle unwrapping" begin
        @test unwrap_angle(3.1, -3.1) ≈ 3.1 - 2π
        @test unwrap_angle(-3.1, 3.1) ≈ -3.1 + 2π
        @test unwrap_angle(0.5, 0.0) == 0.5
    end

    @testset "trajectories and multi-time sample propagation" begin
        p = t1_leo()
        ts = [0.0, p.period / 3, p.period]
        traj = propagate_trajectory(p.model, p.x0, ts)
        @test traj[1] == p.x0
        @test traj[3] ≈ propagate_state(p.model, p.x0, p.period) rtol = 1e-11
        X0 = sample_gaussian(Xoshiro(61), collect(p.x0), scaled_prior(1, 1.0, 1e-3), 5)
        clouds = propagate_samples(p.model, X0, ts[2:end])
        @test clouds[2] ≈ propagate_samples(p.model, X0, p.period) rtol = 1e-11
        @test_throws ArgumentError propagate_trajectory(p.model, p.x0, [2.0, 1.0])
        @test_throws DimensionMismatch propagate_samples(p.model, zeros(5, 2), [1.0])
    end
end
