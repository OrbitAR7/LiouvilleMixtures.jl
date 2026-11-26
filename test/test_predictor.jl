# Leading-order negentropy predictor J ≈ ¼ΣB² + ½ΣB_abcB_bac = ‖κ₃‖²/12.
# Derivation: docs/src/derivations.md.

# volume-preserving 2D map with Σ B_abc B_bac ≠ 0 (T0 cannot test that term):
#   η₁ = x₁ + k(1 - cos x₁),  η₂ = x₂ / (1 + k sin x₁),  det Dη = 1 exactly
twist(x; k) = [x[1] + k * (1 - cos(x[1])), x[2] / (1 + k * sin(x[1]))]

# exact negentropy of twist(x), x ~ N(0, s²I), by I2 with a 2D Gauss-Hermite rule
function twist_negentropy(s, k; n = 80)
    Ξ, W = gauss_hermite_tensor(2, n)
    X = s .* Ξ
    Y = reduce(hcat, (twist(X[:, j]; k) for j in axes(X, 2)))
    return negentropy_from_covariance(quadrature_mean_cov(Y ./ s, W)[2], I(2))
end

@testset "negentropy predictor" begin
    @testset "Edgeworth and covariance forms agree for any B symmetric in (b, c)" begin
        rng = Xoshiro(41)
        for d in (2, 3, 6)
            A = randn(rng, d, d, d)
            B = (A + permutedims(A, (1, 3, 2))) / 2
            @test negentropy_edgeworth(B) ≈ negentropy_leading(B) rtol = 1e-13
            κ = third_cumulant(B)
            @test κ ≈ permutedims(κ, (2, 1, 3)) && κ ≈ permutedims(κ, (1, 3, 2))
        end
    end

    @testset "T0 shear: analytic leading order" begin
        for (c, σ1, σ2) in ((0.2, 0.3, 0.2), (1.0, 0.3, 0.2), (2.0, 0.5, 0.2))
            S0 = Diagonal([σ1, σ2])
            B = whitened_tensor(x -> shear_map(x; c), zeros(2), S0)
            # only B₂₁₁ = 2cσ1²/σ2 is non-zero
            @test B[2, 1, 1] ≈ 2c * σ1^2 / σ2
            @test count(!iszero, B) == 1
            @test volume_trace(B) == zeros(2)
            @test negentropy_leading(B) ≈ t0_negentropy_leading(c, σ1, σ2) rtol = 1e-14
        end
        # the exact J = ½ ln(1 + 2k²) approaches the prediction k² as k → 0
        for σ1 in (0.1, 0.03, 0.01)
            k² = t0_negentropy_leading(1.0, σ1, 0.2)
            @test abs(t0_negentropy(1.0, σ1, 0.2) / k² - 1) <= 1.01 * k²     # = k² + O(k⁴)
        end
    end

    @testset "2D twist map: the B_abc B_bac term is needed, error O(s²)" begin
        k = 0.5
        for s in (0.2, 0.1, 0.05, 0.02)
            B = whitened_tensor(x -> twist(x; k), zeros(2), s * I(2))
            @test maximum(abs, volume_trace(B)) < 1e-14
            J = twist_negentropy(s, k)
            # relative error of the predictor is O(s²); measured about 0.3 s²
            @test abs(J / negentropy_leading(B) - 1) < 0.5 * s^2
            # ¼ΣB² alone is wrong by the factor 7/3 (here B₁₁₁ = ks, B₂₁₂ = B₂₂₁ = -ks)
            s == 0.02 && @test J / (sum(abs2, B) / 4) ≈ 7 / 3 rtol = 2e-3
        end
        # the other limit: unit prior, weakening map (k → 0). Higher Taylor terms of
        # the map enter at the same order, and the ratio tends to 3/7, not 1.
        k = 0.01
        B = whitened_tensor(x -> twist(x; k), zeros(2), I(2))
        @test twist_negentropy(1.0, k) / negentropy_leading(B) ≈ 3 / 7 rtol = 5e-3
    end

    @testset "orbit tensors satisfy the volume constraint Σₐ B_aac = 0" begin
        for (p, P0) in ((t1_leo(), scaled_prior(1, 1.0, 1e-3)),
                        (t1_molniya(), scaled_prior(1, 1.0, 1e-3)),
                        (NRHO, scaled_prior(1, 10 / NRHO_L, 1e-4 / NRHO_V)))
            sol = propagate_tensors(p.model, p.x0, p.period)
            B = whitened_tensor(sol.Φ[end], sol.Ψ[end], sqrt(P0))
            # independent of how Ψ was computed: it follows from ∂ ln det Φ/∂x₀ = 0
            @test maximum(abs, volume_trace(B)) <= 1e-9 * norm(B)
        end
    end

    @testset "predictor does not depend on the square root of P0" begin
        p = t1_leo()
        P0 = [1.0 0.3 0 0 0 0; 0.3 2.0 0 0 0 0; 0 0 1.5 0 0 0;
              0 0 0 1e-6 0 0; 0 0 0 0 2e-6 1e-7; 0 0 0 0 1e-7 1e-6] .* 0.01
        J1 = predicted_negentropy(p.model, p.x0, P0, p.period)
        J2 = predicted_negentropy(p.model, p.x0, P0, p.period; S0 = sqrt(Symmetric(P0)))
        @test J1 ≈ J2 rtol = 1e-10
        sol = propagate_tensors(p.model, p.x0, p.period; order = 1)
        @test symplectic_inverse(p.model, sol.Φ[end]) * sol.Φ[end] ≈ I atol = 1e-9
    end

    # Orbit problems against the deterministic Gauss-Hermite reference (5⁶ nodes,
    # converged: n = 7 agrees to 7 digits). Priors are s × (1 km, 1 m/s) for T1 and
    # s × (10 km, 10 cm/s) for the NRHO.
    @testset "predictor within 10% of the exact negentropy in the small-prior limit" begin
        cases = ((t1_leo(), 1.0, 1.0, 1e-3, 0.01),
                 (t1_leo(J2Gravity()), 1.0, 1.0, 1e-3, 0.01),
                 (t1_molniya(), 0.5, 1.0, 1e-3, 0.03),
                 (t1_molniya(), 0.25, 1.0, 1e-3, 0.1),
                 (NRHO, 1.0, 10 / NRHO_L, 1e-4 / NRHO_V, 1.0))
        ratio_leo_001 = NaN
        for (p, frac, σr, σv, s) in cases
            P0 = scaled_prior(s, σr, σv)
            t = frac * p.period
            ratio = gh_negentropy(p, t, P0) / predicted_negentropy(p.model, p.x0, P0, t)
            p.name == "T1 LEO" && p.model isa TwoBody && (ratio_leo_001 = ratio)
            @test abs(ratio - 1) < 0.1          # target 10%; measured below 0.1%
            @test abs(ratio - 1) < 1e-3
        end
        # the error grows as s²: from s = 0.01 to 0.03 it grows about 9-fold on LEO
        p = t1_leo()
        err(s) = 1 - gh_negentropy(p, p.period, scaled_prior(s, 1.0, 1e-3)) /
                     predicted_negentropy(p.model, p.x0, scaled_prior(s, 1.0, 1e-3), p.period)
        e1, e3 = 1 - ratio_leo_001, err(0.03)
        @test 7 < e3 / e1 < 11
        # where it fails: at the 1 km / 1 m/s prior the predictor overestimates
        # the LEO negentropy about sixfold (8.3 predicted, 1.44 exact)
        @test 0.1 < 1 - err(1.0) < 0.25
    end

    FULL_TESTS && @testset "predictor against Monte Carlo on T1 LEO (s = 0.03)" begin
        # paired estimator in whitened coordinates, N = 2e5; the 99% interval is
        # about ±14% of J, so this checks consistency at Monte Carlo resolution;
        # the Gauss-Hermite test above is the precise one
        p = t1_leo()
        P0 = scaled_prior(0.03, 1.0, 1e-3)
        S0 = sqrt(P0)
        X0 = sample_gaussian(Xoshiro(2030), collect(p.x0), P0, 200_000)
        Xt = propagate_samples(p.model, X0, p.period)
        sol = propagate_tensors(p.model, p.x0, p.period; order = 1)
        H = S0 \ (symplectic_inverse(p.model, sol.Φ[end]) * (Xt .- sol.x[end]))
        e = negentropy_mc(S0 \ (X0 .- p.x0), H, I(6); B = 300, level = 0.99, rng = Xoshiro(10))
        Jp = predicted_negentropy(p.model, p.x0, P0, p.period)
        @test e.lo > 0
        @test e.lo <= 1.1Jp && 0.9Jp <= e.hi       # the ±10% band meets the interval
    end
end
