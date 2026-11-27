# Mixtures, measurement models and the propagation policies P0-P5.

import EstimationToolkit as ETK

# linear dynamics ẋ = A x, for checks where every moment-matching rule is exact
struct LinearModel <: DynamicsModel
    A::SMatrix{6,6,Float64,36}
end
LiouvilleMixtures.vector_field(m::LinearModel, x::SVector{6}) = m.A * x

# identity measurement: reduces the measurement-space criterion to the state one
struct IdentityMeasurement <: MeasurementModel
    σ::Float64
end
LiouvilleMixtures.measure(::IdentityMeasurement, x::AbstractVector) = SVector{6}(x[1], x[2], x[3], x[4], x[5], x[6])
LiouvilleMixtures.measurement_cov(m::IdentityMeasurement) = SMatrix{6,6,Float64}(m.σ^2 * I)

# EstimationToolkit's adaptive mixture filter with our dynamics, for the P2 cross-check
struct ETKModel{M} <: ETK.AbstractDynamicsModel{6,0,6}
    model::M
end
ETK.propagate(w::ETKModel, x, u, dt) = propagate_state(w.model, x, dt)
ETK.process_noise_cov(::ETKModel, dt) = zeros(6, 6)

random_spd(rng, d) = (A = randn(rng, d, d); Symmetric(A * A' + I))

function moments_close(a, b; rtol)
    μa, Σa = a
    μb, Σb = b
    return norm(μa - μb) <= rtol * norm(μb) + rtol * sqrt(norm(Σb)) && norm(Σa - Σb) <= rtol * norm(Σb)
end

@testset "mixtures, measurements and policies" begin
    leo = t1_leo()
    P0_leo = scaled_prior(1, 1.0, 1e-3)
    prior_leo = GaussianMixture(leo.x0, P0_leo)
    angles = Angles(zeros(3), 1e-5)
    ts = [leo.period / 4, leo.period / 2, leo.period]
    epochs = [leo.period / 2]

    @testset "mixture basics" begin
        rng = Xoshiro(51)
        gm = GaussianMixture([2.0, 6.0], [randn(rng, 6), randn(rng, 6)], [random_spd(rng, 6), random_spd(rng, 6)])
        @test gm.w ≈ [0.25, 0.75] && ncomponents(gm) == 2
        @test_throws ArgumentError GaussianMixture([-1.0, 2.0], [zeros(6), zeros(6)], [I(6), I(6)])
        @test_throws DimensionMismatch GaussianMixture([1.0], [zeros(6), zeros(6)], [I(6)])
        g1 = GaussianMixture(leo.x0, P0_leo)
        @test all(mixture_mean_cov(g1) .≈ (leo.x0, P0_leo))
        X = randn(rng, 6, 5)
        direct = [log(gm.w[1] * exp(gaussian_logpdf(X[:, j], gm.m[1], gm.P[1])) +
                      gm.w[2] * exp(gaussian_logpdf(X[:, j], gm.m[2], gm.P[2]))) for j in 1:5]
        @test mixture_logpdf(gm, X) ≈ direct rtol = 1e-12
        @test all(isfinite, mixture_logpdf(gm, fill(1e3, 6, 1)))   # no underflow to -Inf
    end

    @testset "splits preserve mean and covariance to 1e-10" begin
        rng = Xoshiro(52)
        for _ in 1:20
            gm = GaussianMixture(rand(rng, 3), [randn(rng, 6) for _ in 1:3], [random_spd(rng, 6) for _ in 1:3])
            i = rand(rng, 1:3)
            gs = split_component(gm, i, randn(rng, 6); shrink = rand(rng) * 0.8 + 0.1)
            @test ncomponents(gs) == 5
            @test sum(gs.w) ≈ 1 atol = 1e-14
            @test moments_close(mixture_mean_cov(gs), mixture_mean_cov(gm); rtol = 1e-10)
        end
        # on an orbit-scale component (km and km/s, very different magnitudes)
        gs = split_component(prior_leo, 1, [0, 0, 0, 0, 1.0, 0])
        @test moments_close(mixture_mean_cov(gs), mixture_mean_cov(prior_leo); rtol = 1e-10)
    end

    @testset "merges preserve mean and covariance; Runnalls bound" begin
        rng = Xoshiro(53)
        gm = GaussianMixture(rand(rng, 4), [randn(rng, 6) for _ in 1:4], [random_spd(rng, 6) for _ in 1:4])
        gmm = merge_components(gm, 2, 4)
        @test ncomponents(gmm) == 3 && sum(gmm.w) ≈ 1
        @test moments_close(mixture_mean_cov(gmm), mixture_mean_cov(gm); rtol = 1e-12)
        @test merge_cost(gm, 1, 2) > 0
        same = GaussianMixture([1.0, 1.0], [gm.m[1], gm.m[1]], [gm.P[1], gm.P[1]])
        @test abs(merge_cost(same, 1, 2)) < 1e-12
        @test_throws ArgumentError merge_components(gm, 2, 2)
        # P5's merge step: a huge threshold merges everything, keeping the moments;
        # a zero threshold merges nothing
        all1 = LiouvilleMixtures._merge_redundant(gm, 1e6)
        @test ncomponents(all1) == 1
        @test moments_close(mixture_mean_cov(all1), mixture_mean_cov(gm); rtol = 1e-12)
        @test ncomponents(LiouvilleMixtures._merge_redundant(gm, 0.0)) == 4
        # merging the three children of a split gives back the original component
        gs = split_component(prior_leo, 1, [1.0, 0, 0, 0, 0, 0])
        back = merge_components(merge_components(gs, 1, 2), 1, 2)
        @test ncomponents(back) == 1
        @test moments_close(mixture_mean_cov(back), mixture_mean_cov(prior_leo); rtol = 1e-10)
    end

    @testset "measurement models" begin
        x = SVector(7000.0, 100.0, -50.0, 0.1, 7.5, 0.2)
        rr = RangeRangeRate(SVector(10.0, 0.0, 0.0), 0.01, 1e-5)
        d = x[1:3] - rr.origin
        @test measure(rr, x) ≈ [norm(d), dot(d, x[4:6]) / norm(d)]
        an = Angles(zeros(3), 1e-5)
        @test measure(an, x) ≈ [atan(100.0, 7000.0), asin(-50.0 / norm(x[1:3]))]
        @test measurement_cov(rr) ≈ Diagonal([1e-4, 1e-10])
        # derivatives against central differences
        for mm in (rr, an)
            H = measurement_jacobian(mm, x)
            H2 = measurement_hessian(mm, x)
            for k in 1:6
                h = 1e-4 * max(1, abs(x[k]))
                e = SVector{6}(ntuple(i -> i == k ? h : 0.0, 6))
                @test (measure(mm, x + e) - measure(mm, x - e)) / 2h ≈ H[:, k] rtol = 1e-6 atol = 1e-12
                @test (measurement_jacobian(mm, x + e) - measurement_jacobian(mm, x - e)) / 2h ≈ H2[:, :, k] rtol = 1e-5 atol = 1e-12
            end
        end
    end

    @testset "measurement criterion reduces to the state criterion" begin
        # h = identity and R → 0: the whitening differs by an orthogonal matrix,
        # and the norm of κ₃ is invariant under it
        sol = propagate_tensors(leo.model, leo.x0, leo.period)
        S0 = Matrix(cholesky(Symmetric(Matrix(scaled_prior(0.1, 1.0, 1e-3)))).L)
        Jstate = negentropy_leading(whitened_tensor(sol.Φ[end], sol.Ψ[end], S0))
        A, Q, _ = measurement_tensors(IdentityMeasurement(1e-9), sol.Φ[end], sol.Ψ[end], sol.x[end], S0)
        @test measurement_negentropy(A, Q) ≈ Jstate rtol = 1e-6
        # large noise swamps the non-Gaussian part
        A, Q, _ = measurement_tensors(IdentityMeasurement(1e3), sol.Φ[end], sol.Ψ[end], sol.x[end], S0)
        @test measurement_negentropy(A, Q) < 1e-6 * Jstate
    end

    @testset "measurement criterion against the third cumulant by quadrature" begin
        # z = h(φ(x₀)) + v with x₀ on a 5⁶ Gauss-Hermite rule; the noise adds only
        # covariance. The tensor formula is leading order, so compare at modest
        # priors. (Angles are avoided near azimuth ±π, where atan's branch cut
        # breaks any sample-based moment.)
        function kappa_direct(meas, P0, h)
            Ξ, W = gauss_hermite_tensor(6, 5)
            S0 = Matrix(cholesky(Symmetric(Matrix(P0))).L)
            Y = propagate_samples(leo.model, leo.x0 .+ S0 * Ξ, h)
            Z = reduce(hcat, [measure(meas, Y[:, j]) for j in axes(Y, 2)])
            D = Z .- Z * W
            Wh = inv(cholesky(Symmetric((D .* W') * D' + Matrix(measurement_cov(meas)))).L)
            Dw = Wh * D
            κ = [sum(W .* Dw[a, :] .* Dw[b, :] .* Dw[c, :]) for a in 1:2, b in 1:2, c in 1:2]
            return sum(abs2, κ) / 12
        end
        for (meas, s) in ((Angles(zeros(3), 1e-6), 1.0),
                          (RangeRangeRate(SVector(0.0, 6000.0, 3000.0), 1e-4, 1e-8), 1.0),
                          (RangeRangeRate(zeros(3), 1e-4, 1e-8), 0.1))
            P0 = scaled_prior(s, 1.0, 1e-3)
            sol = propagate_tensors(leo.model, leo.x0, leo.period)
            S0 = Matrix(cholesky(Symmetric(Matrix(P0))).L)
            A, Q, _ = measurement_tensors(meas, sol.Φ[end], sol.Ψ[end], sol.x[end], S0)
            @test measurement_negentropy(A, Q) ≈ kappa_direct(meas, P0, leo.period) rtol = 1e-3
        end
    end

    @testset "direction of largest nonlinearity" begin
        # T0: only B₂₁₁ is non-zero, so the direction is the first axis
        B = whitened_tensor(x -> shear_map(x; c = 1.0), zeros(2), Diagonal([0.3, 0.2]))
        u, f = max_nonlinearity_direction(B)
        @test abs(u[1]) ≈ 1 && f ≈ B[2, 1, 1]^2
        # a rank-one stack T_c = a_c v vᵀ bends most along v
        rng = Xoshiro(54)
        v = normalize(randn(rng, 6))
        T = permutedims(cat([a * v * v' for a in randn(rng, 3)]...; dims = 3), (3, 1, 2))
        u, _ = max_nonlinearity_direction(T)
        @test abs(dot(u, v)) ≈ 1 atol = 1e-8
        # never worse than the best coordinate axis or a random start
        T = randn(rng, 4, 6, 6)
        T = (T + permutedims(T, (1, 3, 2))) / 2
        u, f = max_nonlinearity_direction(T)
        fval(w) = sum((dot(w, T[c, :, :] * w))^2 for c in 1:4)
        @test f ≈ fval(u) && all(f >= fval(normalize(randn(rng, 6))) - 1e-12 for _ in 1:200)
    end

    @testset "every policy reduces to P0 when its thresholds are infinite" begin
        ref = propagate_policy(LinearPropagation(), leo.model, prior_leo, ts)
        for pol in (EntropyTrigger(Inf; dt = leo.period / 8),
                    HorizonSplit(Inf),
                    HorizonSplit(Inf; measurement = angles),
                    HorizonSplit(Inf; measurement = angles, τ_merge = 1e-3))
            out = propagate_policy(pol, leo.model, prior_leo, ts; epochs)
            @test all(ncomponents.(out) .== 1)
            @test all(moments_close(mixture_mean_cov(o), mixture_mean_cov(r); rtol = 1e-10) for (o, r) in zip(out, ref))
        end
        # P1 has a cadence, not a threshold. On linear dynamics every refit rule is
        # exact, so it must equal P0 for any cadence
        A = SMatrix{6,6}([zeros(3, 3) I; -I zeros(3, 3)] + 0.1 * randn(Xoshiro(55), 6, 6))
        lin = LinearModel(A)
        g0 = GaussianMixture(ones(6), Matrix(scaled_prior(1, 1.0, 0.5)))
        tl = [0.5, 1.3, 2.0]
        ref = propagate_policy(LinearPropagation(), lin, g0, tl)
        for pol in (ScheduledRefit(0.4), ScheduledRefit(0.7; phase = 0.1), ScheduledRefit(Inf),
                    ScheduledRefit(0.5; method = :quadrature, n_quad = 2))
            out = propagate_policy(pol, lin, g0, tl)
            @test all(moments_close(mixture_mean_cov(o), mixture_mean_cov(r); rtol = 1e-10) for (o, r) in zip(out, ref))
        end
    end

    @testset "P1 with Δ = ∞ is one transform over the whole horizon" begin
        out = propagate_policy(ScheduledRefit(Inf), leo.model, prior_leo, [leo.period])
        Ξ, wm, wc, _ = LiouvilleMixtures._sigma_rule(1.0, 0.0, 0.0)
        X = leo.x0 .+ Matrix(sqrt(Symmetric(Matrix(P0_leo)))) * Ξ
        μ, P = LiouvilleMixtures._rule_moments(propagate_samples(leo.model, X, leo.period), wm, wc)
        @test moments_close(mixture_mean_cov(out[1]), (μ, P); rtol = 1e-10)
    end

    @testset "P1 exact refits never lose covariance volume (ratchet, I3)" begin
        # a 3⁶-point Gauss-Hermite refit every quarter revolution, on LEO
        tq = leo.period .* (1:8) ./ 8
        out = propagate_policy(ScheduledRefit(leo.period / 4; method = :quadrature, n_quad = 3),
                               leo.model, prior_leo, tq)
        ld = [logdet(Symmetric(Matrix(mixture_mean_cov(o)[2]))) for o in out[2:2:end]]   # at the refits
        @test all(diff(vcat(logdet(Symmetric(Matrix(P0_leo))), ld)) .>= -1e-9)
    end

    @testset "splitting policies: weights, component cap, merging (P2-P5)" begin
        cases = (("P2", EntropyTrigger(0.02; dt = leo.period / 8, max_components = 9)),
                 ("P3", HorizonSplit(0.05; max_components = 9)),
                 ("P4", HorizonSplit(1e-7; measurement = angles, max_components = 9)),
                 ("P5", HorizonSplit(1e-7; measurement = angles, τ_merge = 1e-2, max_components = 9)))
        results = Dict{String,Vector{GaussianMixture}}()
        for (name, pol) in cases
            out = propagate_policy(pol, leo.model, prior_leo, ts; epochs)
            results[name] = out
            @test all(abs(sum(o.w) - 1) < 1e-14 for o in out)
            @test all(1 <= ncomponents(o) <= 9 for o in out)
            @test ncomponents(out[end]) > 1          # the thresholds are low enough to split
        end
        # P5 with merging switched off is P4, component for component
        p4 = results["P4"]
        p5 = propagate_policy(HorizonSplit(1e-7; measurement = angles, τ_merge = 0.0, max_components = 9), leo.model, prior_leo, ts; epochs)
        @test all(o4.w ≈ o5.w && o4.m ≈ o5.m && o4.P ≈ o5.P for (o4, o5) in zip(p4, p5))
        # a large merge threshold collapses the mixture at the epoch before it re-splits
        if FULL_TESTS
            p5big = propagate_policy(HorizonSplit(1e-7; measurement = angles, τ_merge = 1e6, max_components = 9), leo.model, prior_leo, ts; epochs)
            @test all(abs(sum(o.w) - 1) < 1e-14 for o in p5big)
        end
    end

    @testset "P3/P4 decisions: split until every component is within τ" begin
        # over half a revolution (a full one needs more than 50 components at these τ)
        for (pol, τ) in ((HorizonSplit(0.01; max_components = 50), 0.01),
                         (HorizonSplit(5e-7; measurement = angles, max_components = 50), 5e-7))
            gm, crit = LiouvilleMixtures._decide_splits(pol, leo.model, prior_leo, leo.period / 2)
            @test 1 < ncomponents(gm) < 50                 # split, but not up to the cap
            @test all(crit .<= τ)
            @test moments_close(mixture_mean_cov(gm), mixture_mean_cov(prior_leo); rtol = 1e-10)
        end
        # with a cap it stops there, and the worst component stays above τ
        gm, crit = LiouvilleMixtures._decide_splits(HorizonSplit(1e-4; max_components = 7), leo.model, prior_leo, leo.period)
        @test ncomponents(gm) == 7 && maximum(crit) > 1e-4
    end

    @testset "P2 step measure and direction agree with EstimationToolkit" begin
        # the same cubature rule and central-difference linearization, written
        # independently; EstimationToolkit propagates with sigma points, so only
        # the first step's measure is comparable
        h = leo.period / 8
        ΔH, s = LiouvilleMixtures._entropy_step(leo.model, leo.x0, P0_leo, h)
        f = ETK.AdaptiveGaussianMixtureFilter(leo.x0, SMatrix{6,6}(Matrix(P0_leo)); split_tol = Inf)
        ETK.predict!(f, ETKModel(leo.model), nothing, h)
        @test ΔH ≈ f.entropy_dev[1] rtol = 1e-8
        @test ΔH > 0
        rule = ETK._sigma_rule(f)
        r = ETK._sigma_transform(x -> propagate_state(leo.model, x, h), leo.x0, SMatrix{6,6}(Matrix(P0_leo)), rule)
        @test s ≈ r.S[:, ETK._most_nonlinear_direction(r)] rtol = 1e-8
    end

    @testset "policies on the NRHO" begin
        # exercises the CR3BP path: range and range-rate from the Earth
        P0n = scaled_prior(10, 10 / NRHO_L, 1e-4 / NRHO_V)        # 100 km, 1 m/s
        prior = GaussianMixture(NRHO.x0, P0n)
        rr = RangeRangeRate(SVector(-NRHO.model.system.μ, 0.0, 0.0), 0.01 / NRHO_L, 1e-6 / NRHO_V)
        tn = [NRHO.period / 2, NRHO.period]
        for pol in (HorizonSplit(0.01; max_components = 7), HorizonSplit(0.01; measurement = rr, max_components = 7))
            out = propagate_policy(pol, NRHO.model, prior, tn; epochs = [NRHO.period / 2])
            @test all(abs(sum(o.w) - 1) < 1e-14 for o in out)
            @test all(ncomponents(o) <= 7 for o in out)
        end
    end

    @testset "argument checks" begin
        @test_throws ArgumentError ScheduledRefit(0.0)
        @test_throws ArgumentError ScheduledRefit(1.0; method = :other)
        @test_throws ArgumentError EntropyTrigger(0.1; dt = 0.0)
        @test_throws ArgumentError propagate_policy(LinearPropagation(), leo.model, prior_leo, [2.0, 1.0])
        two = split_component(prior_leo, 1, ones(6))
        @test_throws ArgumentError propagate_policy(ScheduledRefit(100.0), leo.model, two, [10.0])
    end
end
