# Gaussian-sum measurement update, importance-weighted posterior KL, the drag
# model and its flow divergence.

# linear measurement z = Hm x + v, where the extended Kalman update is exact
struct LinearMeasurement <: MeasurementModel
    Hm::Matrix{Float64}
    R::Matrix{Float64}
end
LiouvilleMixtures.measure(m::LinearMeasurement, x::AbstractVector) = m.Hm * x
LiouvilleMixtures.measurement_cov(m::LinearMeasurement) = m.R

# KL(N(μa, Σa) ‖ N(μb, Σb)), closed form
function gauss_kl(μa, Σa, μb, Σb)
    d = length(μa)
    return (tr(Σb \ Σa) + dot(μb - μa, Σb \ (μb - μa)) - d + logdet(Σb) - logdet(Σa)) / 2
end

@testset "measurement update, posterior KL and drag" begin
    rng = Xoshiro(71)
    Hm = randn(rng, 2, 6)
    R = [0.5 0.1; 0.1 0.3]
    lm = LinearMeasurement(Hm, R)

    @testset "single-Gaussian update is the Kalman filter" begin
        m = randn(rng, 6)
        A = randn(rng, 6, 6); P = A * A' + I
        z = randn(rng, 2)
        g = measurement_update(GaussianMixture(m, P), lm, z)
        S = Hm * P * Hm' + R
        K = P * Hm' / S
        @test g.m[1] ≈ m + K * (z - Hm * m) rtol = 1e-12
        @test g.P[1] ≈ (I - K * Hm) * P rtol = 1e-10
        @test g.w == [1.0]
    end

    @testset "mixture weights follow the predicted likelihoods" begin
        gm = GaussianMixture([0.2, 0.5, 0.3], [randn(rng, 6) for _ in 1:3], [Matrix(1.0I, 6, 6) * k for k in 1:3])
        z = randn(rng, 2)
        g = measurement_update(gm, lm, z)
        lik = [gm.w[i] * exp(gaussian_logpdf(z, Hm * gm.m[i], Hm * gm.P[i] * Hm' + R)) for i in 1:3]
        @test g.w ≈ lik / sum(lik) rtol = 1e-12
        @test sum(g.w) ≈ 1
        # a measurement far from all components does not underflow
        gfar = measurement_update(gm, lm, [1e4, -1e4])
        @test all(isfinite, gfar.w) && sum(gfar.w) ≈ 1
    end

    @testset "angle residuals wrap" begin
        an = Angles(zeros(3), 1e-3)
        @test residual(an, [π - 0.01, 0.2], [-π + 0.01, 0.1]) ≈ [-0.02, 0.1]
        @test residual(Range(zeros(3), 1.0), [5.0], [3.0]) == [2.0]
        x = SVector(3.0, 4.0, 0.0, 1.0, 0.0, 0.0)
        @test measure(Range(zeros(3), 1.0), x) ≈ [5.0]
        @test measure(RangeRate(zeros(3), 1.0), x) ≈ [0.6]
        @test measurement_cov(RangeRate(zeros(3), 0.5)) ≈ fill(0.25, 1, 1)
        X = randn(rng, 6, 4)
        rr = RangeRangeRate(SVector(0.1, 0.2, 0.3), 0.5, 0.2)
        z = measure(rr, X[:, 1]) + SVector(0.1, -0.1)
        @test measurement_loglik(rr, z, X) ≈ [gaussian_logpdf(z - measure(rr, X[:, j]), zeros(2), Matrix(measurement_cov(rr))) for j in 1:4]
    end

    @testset "posterior KL by importance weighting (linear-Gaussian truth)" begin
        # identity flow: prior samples are the propagated samples
        m = zeros(6); P = Matrix(Diagonal([1.0, 2.0, 0.5, 1.5, 1.0, 0.7]))
        X = sample_gaussian(Xoshiro(72), m, P, 200_000)
        z = Hm * [0.3, -0.2, 0.1, 0.0, 0.5, -0.4] + [0.2, -0.1]
        lp0 = gaussian_logpdf(X, m, P)
        ll = measurement_loglik(lm, z, X)
        post = measurement_update(GaussianMixture(m, P), lm, z)
        kl0, ess = posterior_kl(lp0, ll, gaussian_logpdf(X, post.m[1], post.P[1]))
        @test ess > 1e4
        # q is the exact posterior: every term equals ln p(z), so the estimate is
        # ln p(z) - ln p̂(z), whose standard error is about 1/√ESS
        @test abs(kl0) < 3 / sqrt(ess)
        # q = the prior: KL(posterior ‖ prior) in closed form
        klp, _ = posterior_kl(lp0, ll, lp0)
        @test klp ≈ gauss_kl(post.m[1], post.P[1], m, P) rtol = 0.05
        @test_throws DimensionMismatch posterior_kl(lp0, ll[1:3], lp0)
    end

    @testset "exponential drag: vector field and divergence" begin
        p = t3_leo_drag()
        m = p.model
        x = p.x0
        @test atmospheric_density(m, R_EARTH + 400.0) ≈ 3.725e-12
        m0 = ExponentialDrag(; ρ0 = 3.725e-12, h0 = 400.0, H = 58.515, β = 0.0)
        @test vector_field(m0, x) ≈ vector_field(TwoBody(), x) rtol = 1e-15
        # analytic divergence: ∇·f = -4·10³ β ρ |v|
        @test flow_divergence(m, x) ≈ -4e3 * m.β * atmospheric_density(m, norm(x[1:3])) * norm(x[4:6]) rtol = 1e-10
        @test abs(flow_divergence(TwoBody(), x)) < 1e-18
        # the orbit loses energy
        E(x) = dot(x[4:6], x[4:6]) / 2 - m.μ / norm(x[1:3])
        traj = propagate_trajectory(m, x, collect(range(0, 5p.period; length = 6)))
        @test issorted(E.(traj); rev = true)
        # Liouville's formula: ln det Φ(t) = ∫₀ᵗ ∇·f(x(s)) ds along the trajectory
        T = p.period
        ts = collect(range(0, T; length = 4001))
        tr_ = propagate_trajectory(m, x, ts)
        d = flow_divergence.(Ref(m), tr_)
        integral = sum((d[1:end-1] .+ d[2:end]) ./ 2 .* diff(ts))
        ld = logdet(propagate_tensors(m, x, T; order = 1).Φ[end])
        @test ld ≈ integral rtol = 1e-6
        @test ld < 0
    end
end
