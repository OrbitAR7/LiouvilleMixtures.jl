# E4: prediction-only accuracy. KL(p_t ‖ q_t) between the true propagated
# density and each policy's approximation, at the decision epoch, against the
# number of mixture components.
#
#   T2: 9:2 NRHO, prior 1 × (10 km, 10 cm/s) at apolune, epochs at the first
#       perilune (T/2) and at apolune (T). P4 uses Earth range and range-rate
#       (σ = 10 m and 1 mm/s).
#   T1: LEO, prior 1 km / 1 m/s, epoch after one revolution. P4 uses
#       geocentric angles (σ = 10 µrad).
#
# Accuracy against size, with no threshold tuned for any reported case:
#   P3 and P4 run with τ = 0 and a cap K. They then split greedily, worst
#     component first by their own criterion and direction, until they reach K.
#     That is the same sequence of splits a decreasing τ would produce.
#   P2 cannot be compared that way: with τ = 0 it splits everything at its first
#     step, before any nonlinearity has built up. So its threshold is swept
#     (τ = 1e-7 … 1, cap 31), and each run is plotted at the number of components
#     it ended with. P2 checks its trigger every T/32.
#   P5 adds merging at decision epochs. With the single decision epoch used here
#     it equals P4, so it is left to E5.
#   P0 and the single-Gaussian P1 (cubature refit every quarter of the horizon)
#     appear at K = 1.
#
# KL is estimated with the exact prior log-density and no density estimation,
# KL = E[ln p₀(x₀) - ln q(φ(x₀))], from N = 2×10⁵ samples, with a 95% normal
# interval. Everything is evaluated in whitened coordinates
# η = S₀⁻¹ Φ⁻¹ (x - x̄), where KL is unchanged. The moment-matched single
# Gaussian has KL = J, the best any single Gaussian can do.
#
#   julia --project=scripts -t auto scripts/e4_accuracy.jl

@isdefined(FIGDIR) || include(joinpath(@__DIR__, "common.jl"))

const E4_SEED = 404
const E4_CAPS = [1, 3, 5, 9, 15, 21, 31]
const E4_TAUS = 10.0 .^ (-7:0)

# Whitened KL of mixture `gm` against samples whose prior whitened coordinates
# are Ξ and whose propagated whitened coordinates are H. W and x̄ define the map.
function whitened_kl(gm::GaussianMixture, W, x̄, Ξ, H)
    gw = GaussianMixture(gm.w, [W * (m - x̄) for m in gm.m], [W * P * W' for P in gm.P])
    lp0 = gaussian_logpdf(Ξ, zeros(6), Matrix(1.0I, 6, 6))
    return kl_mc(lp0, mixture_logpdf(gw, H))
end

function e4_case(name, p, P0, t, meas; N = 200_000)
    S0 = Matrix(cholesky(Symmetric(Matrix(P0))).L)
    X0 = sample_gaussian(Xoshiro(E4_SEED), collect(p.x0), P0, N)
    Ξ = S0 \ (X0 .- p.x0)
    Xt = propagate_samples(p.model, X0, t)
    lin = propagate_tensors(p.model, p.x0, t; order = 1)
    W = S0 \ symplectic_inverse(p.model, lin.Φ[end])
    x̄ = lin.x[end]
    H = W * (Xt .- x̄)
    J = logdet_sample_cov(H) / 2 - logdet_sample_cov(Ξ) / 2       # negentropy (paired)
    prior = GaussianMixture(p.x0, P0)
    kl(pol) = whitened_kl(propagate_policy(pol, p.model, prior, [t])[1], W, x̄, Ξ, H)
    out = Dict{String,Any}()
    out["J"] = J
    out["P0"] = kl(LinearPropagation())
    out["P1"] = kl(ScheduledRefit(t / 4))
    # each entry: (components at the epoch, KL estimate), sorted by components
    run(pol) = (gm = propagate_policy(pol, p.model, prior, [t])[1]; (ncomponents(gm), whitened_kl(gm, W, x̄, Ξ, H)))
    out["P2 entropy trigger"] = [(run(EntropyTrigger(τ; dt = p.period / 32, max_components = 31))..., τ) for τ in E4_TAUS]
    out["P3 horizon, state"] = [run(HorizonSplit(0.0; max_components = K)) for K in E4_CAPS]
    out["P4 horizon, measurement"] = [run(HorizonSplit(0.0; measurement = meas, max_components = K)) for K in E4_CAPS]
    return out
end

function e4_run()
    set_theme_paper!()
    t_start = time()
    nrho = t2_nrho()
    sys = nrho.model.system
    L, V = sys.l_star, sys.l_star / sys.t_star
    P0n = scaled_prior(1.0, 10 / L, 1e-4 / V)
    rr = RangeRangeRate(SVector(-sys.μ, 0.0, 0.0), 0.01 / L, 1e-6 / V)
    leo = t1_leo()
    cases = [("T2 NRHO, first perilune (T/2)", nrho, P0n, nrho.period / 2, rr),
             ("T2 NRHO, apolune (T)", nrho, P0n, nrho.period, rr),
             ("T1 LEO, one revolution", leo, scaled_prior(1.0, 1.0, 1e-3), leo.period, Angles(zeros(3), 1e-5))]
    results = [(name, e4_case(name, p, P0, t, meas)) for (name, p, P0, t, meas) in cases]

    labs = ["P2 entropy trigger", "P3 horizon, state", "P4 horizon, measurement"]
    fig = Figure(size = (900, 400))
    rows = ["| case | policy | KL [95% CI] at K components |", "|---|---|---|"]
    for (ci, (name, r)) in enumerate(results)
        ax = Axis(fig[1, ci]; xscale = log10, yscale = log10, xticks = (E4_CAPS, string.(E4_CAPS)),
                  xlabel = "mixture components K", ylabel = ci == 1 ? "KL to the truth [nats]" : "",
                  title = name, yticks = decade_ticks(-6, 6))
        hlines!(ax, [max(r["J"], 1e-7)]; color = INK2, linestyle = :dash, linewidth = 1,
                label = "moment-matched Gaussian (KL = J)")
        for (i, lab) in enumerate(labs)
            K = [q[1] for q in r[lab]]; v = [q[2] for q in r[lab]]
            if i == 1      # P2: one point per threshold, no connecting line
                scatter!(ax, K, max.([e.value for e in v], 1e-7); color = CAT[i], marker = :diamond, markersize = 8,
                         label = lab * " (τ sweep)")
            else
                lines!(ax, K, max.([e.value for e in v], 1e-7); color = CAT[i], linestyle = STYLES[i], label = lab)
                scatter!(ax, K, max.([e.value for e in v], 1e-7); color = CAT[i], markersize = 7)
            end
        end
        scatter!(ax, [1], [max(r["P0"].value, 1e-7)]; color = INK, marker = :rect, markersize = 9, label = "P0 linear")
        scatter!(ax, [1], [max(r["P1"].value, 1e-7)]; color = INK, marker = :utriangle, markersize = 9, label = "P1 cubature refit")
        ci == 1 && Legend(fig[2, 1:3], ax; orientation = :horizontal, labelsize = 9, tellheight = true)
        push!(rows, @sprintf("| %s | moment-matched Gaussian (= J) | K = 1: %.3g |", name, r["J"]))
        push!(rows, @sprintf("| %s | P0 linear | K = 1: %.3g [%.3g, %.3g] |", name, r["P0"].value, r["P0"].lo, r["P0"].hi))
        push!(rows, @sprintf("| %s | P1 cubature refit | K = 1: %.3g [%.3g, %.3g] |", name, r["P1"].value, r["P1"].lo, r["P1"].hi))
        push!(rows, "| $name | P2 entropy trigger | " * join([@sprintf("τ = %.0e: K = %d, %.3g [%.3g, %.3g]", τ, K, e.value, e.lo, e.hi)
                                                                for (K, e, τ) in r["P2 entropy trigger"]], "; ") * " |")
        for lab in labs[2:3]
            push!(rows, "| $name | $lab | " * join([@sprintf("K = %d: %.3g [%.3g, %.3g]", K, e.value, e.lo, e.hi) for (K, e) in r[lab]], "; ") * " |")
        end
    end
    f = save_figure(fig, "e4_accuracy")

    body = """
    Prediction-only accuracy: KL from the true propagated density to each policy's approximation,
    against the number of components K (95% intervals, N = 2×10⁵). P3 and P4 split greedily to a
    cap K (τ = 0). P2's threshold is swept (τ = 1e-7 … 1, cap 31), and each run is reported at the
    component count it reached. "Moment-matched Gaussian (= J)" is the lowest KL any single
    Gaussian can reach.

    $(join(rows, "\n"))

    ![E4 accuracy]($f)

    *KL to the truth against component count. The dashed line is the moment-matched Gaussian
    (KL = J). Squares mark the linear P0 and triangles the single-Gaussian cubature refit, both
    at K = 1. Lines are the three splitting policies: P3 and P4 split greedily up to K, and P2 is
    shown across its threshold sweep.*
    """
    record!("E4", "prediction-only accuracy against component count", body; seed = E4_SEED, runtime = time() - t_start)
    return results
end

e4_run()
