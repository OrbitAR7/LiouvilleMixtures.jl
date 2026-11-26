# M2: entropy and KL tools. Acceptance tests are marked "M2:".
#
# Statistical tests use fixed seeds (chosen once, before the first run) and
# compare against confidence intervals, never against point values (STANDARDS §4).
# Acceptance tests use 99% intervals. That level was fixed before the suite was
# first run: with about a dozen interval checks, 95% would fail one by chance
# about half the time.

using Distributions: MvNormal, logpdf, entropy

@testset "entropy tools" begin
    @testset "Gaussian log-density and entropy against Distributions.jl" begin
        rng = Xoshiro(11)
        for d in (2, 6)
            A = randn(rng, d, d)
            Σ = A * A' + I
            μ = randn(rng, d)
            X = randn(rng, d, 50)
            ref = MvNormal(μ, Symmetric(Σ))
            @test gaussian_logpdf(X, μ, Σ) ≈ [logpdf(ref, X[:, j]) for j in 1:50] rtol = 1e-12
            @test gaussian_logpdf(X[:, 1], μ, Σ) ≈ logpdf(ref, X[:, 1]) rtol = 1e-12
            @test gaussian_entropy(Σ) ≈ entropy(ref) rtol = 1e-13
        end
        @test_throws Exception gaussian_logpdf(zeros(2, 1), zeros(2), [1.0 2.0; 2.0 1.0])
    end

    @testset "sampling and moments" begin
        rng = Xoshiro(12)
        P = [4.0 1.0; 1.0 2.0]
        X = sample_gaussian(rng, [1.0, -2.0], P, 100_000)
        μ, Σ = sample_mean_cov(X)
        # 5-sigma bounds on the sample mean and covariance (N = 1e5)
        @test all(abs.(μ - [1.0, -2.0]) .< 5 * sqrt.(diag(P) / 1e5))
        @test all(abs.(Σ - P) .< 5 * sqrt.((P .^ 2 .+ diag(P) * diag(P)') / 1e5))
        # weighted moments: unit weights reproduce the plain ones, and integer
        # counts equal replicating the columns
        @test all(sample_mean_cov(X, ones(Int, size(X, 2))) .≈ sample_mean_cov(X))
        Y = randn(rng, 3, 7)
        w = [0, 2, 1, 0, 3, 1, 0]
        Yrep = reduce(hcat, [repeat(Y[:, j], 1, w[j]) for j in 1:7 if w[j] > 0])
        μw, Σw = sample_mean_cov(Y, w)
        μr, Σr = sample_mean_cov(Yrep)
        @test μw ≈ μr && Σw ≈ Σr
    end

    @testset "Gauss-Hermite rule (Golub-Welsch)" begin
        # exact standard-normal moments E[ξᵏ] = (k-1)!! for even k, 0 for odd k,
        # for every k ≤ 2n - 1
        dfact(k) = k <= 0 ? 1 : k * dfact(k - 2)
        for n in (1, 2, 3, 5, 7, 12)
            x, w = gauss_hermite(n)
            @test sum(w) ≈ 1
            @test x ≈ -reverse(x) atol = 1e-13
            for k in 0:2n-1
                @test sum(w .* x .^ k) ≈ (isodd(k) ? 0.0 : Float64(dfact(k - 1))) atol = 1e-11 * max(1, dfact(k - 1))
            end
        end
        @test_throws ArgumentError gauss_hermite(0)
        Ξ, W = gauss_hermite_tensor(3, 4)
        @test size(Ξ) == (3, 64) && sum(W) ≈ 1
        μ, Σ = quadrature_mean_cov(Ξ, W)
        @test norm(μ) < 1e-14 && Σ ≈ I
        # T0 is a degree-2 polynomial map, so its output covariance has degree 4
        # and a 3-point rule is exact
        Ξ, W = gauss_hermite_tensor(2, 3)
        for (c, σ1, σ2) in ((1.0, 0.3, 0.2), (2.0, 0.5, 0.2))
            Y = shear_map(Diagonal([σ1, σ2]) * Ξ; c)
            _, Σy = quadrature_mean_cov(Y, W)
            @test negentropy_from_covariance(Σy, Diagonal([σ1^2, σ2^2])) ≈ t0_negentropy(c, σ1, σ2) rtol = 1e-13
        end
    end

    # T0 cases: small, moderate and large non-Gaussianity
    T0_CASES = ((0.2, 0.3, 0.2), (1.0, 0.3, 0.2), (2.0, 0.5, 0.2))
    N_MC = 200_000

    @testset "M2: T0 exact negentropy inside the Monte Carlo interval (c=$c, σ1=$σ1, σ2=$σ2)" for (c, σ1, σ2) in T0_CASES
        P0 = Diagonal([σ1^2, σ2^2])
        X = sample_gaussian(Xoshiro(2026), zeros(2), P0, N_MC)
        Y = shear_map(X; c)
        J = t0_negentropy(c, σ1, σ2)
        es = negentropy_mc(X, Y, P0; prior = :sample, B = 500, level = 0.99, rng = Xoshiro(7))
        ee = negentropy_mc(X, Y, P0; prior = :exact, B = 500, level = 0.99, rng = Xoshiro(7))
        for e in (es, ee)
            @test J ∈ e
            @test e.n == N_MC && e.level == 0.99 && e.lo < e.value < e.hi
        end
        # the paired estimator (prior = :sample) has the smaller error here: the
        # shared x₁ sampling error cancels in ln det Σ̂_y - ln det Σ̂_x
        @test es.se < ee.se
    end

    @testset "bootstrap interval coverage" begin
        # 200 independent T0 experiments with N = 2000; a 95% interval should
        # contain the truth in about 190 of them (binomial sd ≈ 3). Accept
        # 180–199: percentile intervals for a log-determinant at this N can
        # undercover slightly, but not by more than a few percent.
        c, σ1, σ2 = 1.0, 0.3, 0.2
        P0 = Diagonal([σ1^2, σ2^2])
        J = t0_negentropy(c, σ1, σ2)
        rng = Xoshiro(31)
        hits = count(1:200) do _
            X = sample_gaussian(rng, zeros(2), P0, 2000)
            J ∈ negentropy_mc(X, shear_map(X; c), P0; B = 200, rng)
        end
        @test 180 <= hits <= 199
    end

    @testset "M2: KL estimator is zero when q is the truth (c=$c, σ1=$σ1, σ2=$σ2)" for (c, σ1, σ2) in T0_CASES
        P0 = Diagonal([σ1^2, σ2^2])
        X = sample_gaussian(Xoshiro(2027), zeros(2), P0, N_MC)
        Y = shear_map(X; c)
        lp0 = gaussian_logpdf(X, zeros(2), P0)
        # Each term ln p₀(x) - ln q(φ(x)) is analytically zero, so "within error"
        # here means within rounding. A statistical interval can't express that:
        # the terms are all about -1e-16 with one sign, so their 99% interval is
        # about 1e-18 wide and excludes 0. Test the rounding level instead.
        e = kl_mc(lp0, shear_logpdf(Y; c, σ1, σ2); level = 0.99)
        @test abs(e.value) < 1e-14
        @test e.hi - e.lo < 1e-14

        # non-trivial: the linearized Gaussian N(0, P0) has a closed-form KL ...
        e_lin = kl_mc(lp0, gaussian_logpdf(Y, zeros(2), P0); level = 0.99)
        @test t0_kl_to_prior(c, σ1, σ2) ∈ e_lin
        # ... and the moment-matched Gaussian gives the negentropy itself
        μy, Σy = sample_mean_cov(Y)
        e_mm = kl_mc(lp0, gaussian_logpdf(Y, μy, Σy); level = 0.99)
        @test t0_negentropy(c, σ1, σ2) ∈ e_mm
        # with log |det Dφ| passed explicitly as zeros, nothing changes
        @test kl_mc(lp0, gaussian_logpdf(Y, μy, Σy); logdetjac = zeros(N_MC)).value ≈ e_mm.value
    end

    # T1: prior 1 km and 1 m/s per axis, one revolution, N = 20 000 (the 2e5 of
    # the brief is for the experiments; here it would cost minutes).
    @testset "M2: I2 holds on T1 ($(p.name), $(nameof(typeof(p.model))))" for p in (t1_leo(), t1_leo(J2Gravity()), t1_molniya())
        P0 = Diagonal([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6])          # km², (km/s)²
        X0 = sample_gaussian(Xoshiro(2028), collect(p.x0), P0, 20_000)
        Xt, ldj = propagate_samples(p.model, X0, p.period; logdet = true)
        # the flow is volume-preserving sample by sample
        @test maximum(abs, ldj) < 1e-10
        # I2: ½ ln det Σ_t - ½ ln det P₀ with the exact prior ...
        J_I2 = negentropy_mc(X0, Xt, P0; prior = :exact, B = 300, level = 0.99, rng = Xoshiro(8))
        # ... against KL(p_t ‖ N(μ_t, Σ_t)) by its definition: the exact prior
        # log-density, the per-sample change of variables (ln |det Φ| from each
        # sample's own STM, not assumed zero), and the moment-matched Gaussian
        μt, Σt = sample_mean_cov(Xt)
        KL = kl_mc(gaussian_logpdf(X0, p.x0, P0), gaussian_logpdf(Xt, μt, Σt);
                   logdetjac = ldj, level = 0.99)
        @test J_I2.value ∈ KL
        # and the negentropy is resolved: this is not a 0 ≈ 0 comparison
        @test J_I2.lo > 10 * J_I2.se
    end

    @testset "Monte Carlo covariance against Gauss-Hermite on T1" begin
        # an independent route to Σ_t: a deterministic 5⁶ = 15 625-node rule.
        # The LEO map is not polynomial, so the rule is not exact; at a 0.1 km /
        # 0.1 m/s prior the two agree well within the Monte Carlo interval.
        p = t1_leo()
        P0 = Diagonal([fill(0.01, 3); fill(1e-8, 3)])
        Ξ, W = gauss_hermite_tensor(6, 5)
        Y = propagate_samples(p.model, p.x0 .+ sqrt(P0) * Ξ, p.period)
        J_gh = negentropy_from_covariance(quadrature_mean_cov(Y, W)[2], P0)
        X0 = sample_gaussian(Xoshiro(2029), collect(p.x0), P0, 20_000)
        e = negentropy_mc(X0, propagate_samples(p.model, X0, p.period), P0; B = 300, level = 0.99, rng = Xoshiro(9))
        @test J_gh ∈ e
        @test J_gh > 0.05      # about 0.077 nats: well resolved, not near zero
    end

    @testset "argument checks" begin
        X = zeros(2, 3)
        @test_throws DimensionMismatch negentropy_mc(X, zeros(2, 4), I(2))
        @test_throws ArgumentError negentropy_mc(X, X, I(2); prior = :other)
        @test_throws DimensionMismatch kl_mc(zeros(3), zeros(4))
        @test_throws DimensionMismatch propagate_samples(TwoBody(), zeros(5, 2), 1.0)
    end
end
