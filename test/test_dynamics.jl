# Vector fields and their derivatives: analytic forms against ForwardDiff,
# J2 against the gradient of its potential, and the volume-preservation and
# symplectic structure that M1 relies on.

# random states near each test orbit (fixed seed)
function sample_states(p::TestProblem, n; seed = 1)
    rng = Random.Xoshiro(seed)
    return [p.x0 + 0.05 * p.scales .* SVector{6}(randn(rng, 6)) for _ in 1:n]
end

@testset "dynamics" begin
    problems = (t1_leo(), t1_molniya(), t1_leo(J2Gravity()), NRHO)

    @testset "Jacobian against ForwardDiff: $(p.name), $(nameof(typeof(p.model)))" for p in problems
        for x in sample_states(p, 20)
            A = state_jacobian(p.model, x)
            A_ad = ForwardDiff.jacobian(y -> vector_field(p.model, y), x)
            # analytic and AD agree to rounding; relative to the matrix norm
            @test norm(A - A_ad) <= 1e-13 * norm(A_ad)
            # Liouville: trace A = 0, so det Φ = 1
            @test abs(tr(A)) <= 1e-14 * norm(A)
        end
    end

    @testset "Hessian: $(p.name), $(nameof(typeof(p.model)))" for p in problems
        for x in sample_states(p, 10)
            H = state_hessian(p.model, x)
            H_ad = reshape(ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(z -> vector_field(p.model, z), y)), x), 6, 6, 6)
            @test norm(H - H_ad) <= 1e-12 * norm(H_ad)
            # second derivatives commute
            @test all(H[i, l, m] ≈ H[i, m, l] for i in 1:6, l in 1:6, m in 1:6)
        end
    end

    @testset "J2 acceleration is the gradient of its potential" begin
        m = J2Gravity()
        for x in sample_states(t1_leo(m), 20)
            r = SVector(x[1], x[2], x[3])
            g = ForwardDiff.gradient(y -> potential(m, y), r)
            @test norm(vector_field(m, x)[4:6] - g) <= 1e-14 * norm(g)
        end
        # J2 = 0 reduces to the point mass
        m0 = J2Gravity(GM_EARTH, 0.0, R_EARTH)
        x = t1_molniya().x0
        @test vector_field(m0, x) ≈ vector_field(TwoBody(), x) rtol = 1e-15
        # and J2 is small: about 1e-3 of the central term at LEO
        x = t1_leo().x0
        δa = norm(vector_field(m, x)[4:6] - vector_field(TwoBody(), x)[4:6])
        @test 1e-4 < δa / norm(vector_field(TwoBody(), x)[4:6]) < 3e-3
    end

    @testset "symplectic forms" begin
        J = symplectic_form(TwoBody())
        @test J == symplectic_form(J2Gravity())
        @test J' == -J && J * J == -I
        # CR3BP: Ω = Tᵀ J T for the map T: (r, v) -> (r, p = v + ẑ × r)
        Ω = symplectic_form(CR3BP())
        T = canonical_transform(CR3BP())
        @test Ω == T' * J * T
        @test Ω' == -Ω && abs(det(Ω)) ≈ 1
        x = SVector(0.9, 0.2, 0.1, 0.01, -0.3, 0.05)
        @test (T * x)[4:6] ≈ x[4:6] + cross(SVector(0.0, 0.0, 1.0), x[1:3])
        @test canonical_transform(TwoBody()) == I
    end
end
