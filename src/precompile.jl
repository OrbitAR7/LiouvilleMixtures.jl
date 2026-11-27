# =============================================================================
# Precompile workload
# =============================================================================
#
# Runs the hot paths once at package precompile time, so their compiled code is
# cached with the package rather than recompiled in every session. The cost is
# about 1 s of extra precompilation. The cases are tiny and need no data.

@setup_workload begin
    leo = SVector(7000.0, 0.0, 0.0, 0.0, 7.5, 0.1)
    nrho = SVector(1.0220, 0.0, -0.1821, 0.0, -0.1033, 0.0)    # near the 9:2 NRHO apolune
    P = SMatrix{6,6,Float64}(Diagonal([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6]))
    Pn = SMatrix{6,6,Float64}(Diagonal([1e-9, 1e-9, 1e-9, 1e-8, 1e-8, 1e-8]))
    @compile_workload begin
        for (model, x, Pm, t) in ((TwoBody(), leo, P, 60.0), (J2Gravity(), leo, P, 60.0), (CR3BP(), nrho, Pn, 0.01))
            propagate_state(model, x, t)
            propagate_tensors(model, x, t; order = 1)
            propagate_tensors(model, x, t; order = 1, saveat = [t / 2, t])
            sol = propagate_tensors(model, x, t)
            B = whitened_tensor(sol.Φ[end], sol.Ψ[end], Matrix(cholesky(Symmetric(Matrix(Pm))).L))
            negentropy_leading(B)
            max_nonlinearity_direction(B)
            X0 = sample_gaussian(Random.Xoshiro(1), collect(x), Matrix(Pm), 64)
            Xt, _ = propagate_samples(model, X0, t; logdet = true)
            negentropy_mc(X0, Xt, Matrix(Pm); B = 4, rng = Random.Xoshiro(2))
            gm = GaussianMixture(x, Pm)
            propagate_policy(LinearPropagation(), model, gm, [t])
            propagate_policy(HorizonSplit(0.0; max_components = 3), model, gm, [t])
            propagate_policy(HorizonSplit(0.0; measurement = Angles(zeros(3), 1e-5), max_components = 3), model, gm, [t])
        end
        # drag model and the Gaussian-sum update
        drag = ExponentialDrag(; ρ0 = 3.725e-12, h0 = 400.0, H = 58.515, β = 0.011)
        xd = SVector(6778.137, 0.0, 0.0, 0.0, 7.67, 0.0)
        propagate_state(drag, xd, 60.0)
        propagate_tensors(drag, xd, 60.0; order = 1, saveat = [30.0, 60.0])
        flow_divergence(drag, xd)
        gm = GaussianMixture(leo, P)
        for meas in (Range(zeros(3), 0.01), RangeRate(zeros(3), 1e-5), Angles(zeros(3), 1e-5), RangeRangeRate(zeros(3), 0.01, 1e-5))
            z = measure(meas, leo)
            g = measurement_update(gm, meas, z)
            measurement_loglik(meas, z, reshape(collect(leo), 6, 1))
            mixture_logpdf(g, reshape(collect(leo), 6, 1))
        end
        Ξ, W = gauss_hermite_tensor(2, 3)
        quadrature_mean_cov(shear_map(Ξ; c = 1.0), W)
        kl_mc(zeros(4), zeros(4))
    end
end
