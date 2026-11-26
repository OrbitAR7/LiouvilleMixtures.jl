# M1 acceptance tests. Tolerances are the brief's. Measured values are in
# planning/STATUS.md.
#
# Ψ comparisons use the dimensionless tensor Ψ̃_ijk = Ψ_ijk D_j D_k / D_i
# (D = problem scales: a and √(μ/a) for Kepler/J2, 1 for the CR3BP) and the
# relative Frobenius error. A per-entry relative error is meaningless for
# entries that are zero.

@testset "transition tensors" begin
    problems = (t1_leo(), t1_leo(J2Gravity()), t1_molniya(), t1_molniya(J2Gravity()), NRHO)

    # Central-difference step δ (h_k = δ D_k) for the FD check of Ψ, from a scan
    # over δ = 1e-3 ... 3e-8 on 26 Nov 2026 (relative error at one revolution):
    #   LEO:     falls as δ² down to 5e-10 at 1e-6
    #   Molniya: 2.3e-6 at 1e-6, 2.1e-8 at 1e-7 (strong curvature at perigee)
    #   NRHO:    9e-9 at 3e-6, then roundoff grows (8e-8 at 1e-6, 2e-6 at 3e-8)
    fd_step(p) = p.name == "T1 Molniya" ? 1e-7 : p.name == "T2 NRHO 9:2" ? 3e-6 : 1e-6

    @testset "$(p.name), $(nameof(typeof(p.model)))" for p in problems
        ts = range(0, p.period; length = 41)[2:end]
        sol = propagate_tensors(p.model, p.x0, p.period; saveat = ts)
        @test sol.t ≈ ts
        Φ, Ψ = sol.Φ[end], sol.Ψ[end]

        # M1: det Φ - 1 below 1e-10 over one revolution (40 points along it)
        @test maximum(Φt -> abs(det(Φt) - 1), sol.Φ) < 1e-10

        # M1: symplectic residual below 1e-9, in the form that is correct for
        # the model's coordinates (see symplectic_form), and in the
        # dimensionless state (owner-approved 26 Nov 2026: the brief does not
        # say which units). In km and km/s the Molniya residual has a roundoff
        # floor of 3e-9 to 2e-8 that no tolerance removes (symplectic_residual
        # docstring); LEO passes in either form.
        @test maximum(Φt -> symplectic_residual(p.model, Φt; scales = p.scales), sol.Φ) < 1e-9
        p.name == "T1 LEO" && @test maximum(Φt -> symplectic_residual(p.model, Φt), sol.Φ) < 1e-9

        # Ψ is symmetric in its last two indices. The forcing term is symmetrized
        # in _rhs_order2!, so this holds to rounding of the final scaling only.
        # Accuracy is checked independently against FD and jet transport below.
        Ψs = scaled(Ψ, p.scales)
        @test norm(Ψs - permutedims(Ψs, (1, 3, 2))) <= 1e-14 * norm(Ψs)

        # M1: Ψ against central differences of Φ, 1e-6 relative
        @test relerr(scaled(fd_psi(p, p.period, fd_step(p)), p.scales), Ψs) < 1e-6

        # M1: Ψ against jet transport (TaylorIntegration), 1e-8 relative
        Φj, Ψj, tj = jet_tensors(p, p.period)
        @test tj ≈ p.period
        @test relerr(scaled(Φ, p.scales), scaled(Φj, p.scales)) < 1e-8
        @test relerr(Ψs, scaled(Ψj, p.scales)) < 1e-8

        # the first-order run and the state-only run give the same answers
        s1 = propagate_tensors(p.model, p.x0, p.period; order = 1)
        @test isempty(s1.Ψ)
        @test s1.Φ[end] ≈ Φ rtol = 1e-11
        @test propagate_state(p.model, p.x0, p.period) ≈ sol.x[end] rtol = 1e-11
    end

    @testset "the standard J is the wrong form for the rotating frame" begin
        # documents the correction to the brief's M1 test: ΦᵀJΦ = J fails for the
        # CR3BP in (r, v) even though Φ is exact
        Φ = propagate_tensors(NRHO.model, NRHO.x0, NRHO.period; order = 1).Φ[end]
        J = symplectic_form(TwoBody())
        @test maximum(abs, Φ' * J * Φ - J) > 1
        @test symplectic_residual(NRHO.model, Φ) < 1e-9
    end

    @testset "NRHO reference orbit" begin
        # periodic, 6.5624 d (OrbitalMechanics tests the same value), and the
        # monodromy matrix has the reciprocal eigenvalue pairs of a symplectic map
        sys = NRHO.model.system
        @test norm(propagate_state(NRHO.model, NRHO.x0, NRHO.period) - NRHO.x0) < 1e-9
        @test NRHO.period * sys.t_star / 86400 ≈ 6.5624 atol = 1e-4
        λ = eigvals(propagate_tensors(NRHO.model, NRHO.x0, NRHO.period; order = 1).Φ[end])
        @test prod(λ) ≈ 1 atol = 1e-10
        @test sort(abs.(λ)) .* reverse(sort(abs.(λ))) ≈ ones(6) atol = 1e-6
    end

    @testset "errors" begin
        @test_throws ArgumentError propagate_tensors(TwoBody(), t1_leo().x0, 10.0; order = 3)
        # radial free fall hits the singularity at r = 0 in about 1.2e3 s
        @test_throws ErrorException propagate_state(TwoBody(), [7000.0, 0, 0, 0, 0, 0], 5e3)
        @test_throws ErrorException propagate_tensors(TwoBody(), [7000.0, 0, 0, 0, 0, 0], 5e3; order = 1)
    end
end
