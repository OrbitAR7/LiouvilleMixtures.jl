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
        Ξ, W = gauss_hermite_tensor(2, 3)
        quadrature_mean_cov(shear_map(Ξ; c = 1.0), W)
        kl_mc(zeros(4), zeros(4))
    end
end
