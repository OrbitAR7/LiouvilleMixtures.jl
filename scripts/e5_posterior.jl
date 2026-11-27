# E5: posterior accuracy after one measurement update, by measurement type and
# by where on the orbit the measurement is taken. Tests H5.
#
# 9:2 NRHO, prior 1 × (10 km, 10 cm/s) at apolune. One measurement is taken from
# the Earth's centre at one of ten epochs along the first revolution. It is one
# of: range, range-rate, or angles (azimuth and elevation).
#
# Ground truth. The exact posterior is represented by N = 5×10⁵ prior samples,
# propagated exactly and weighted by the likelihood (brief §5). The accuracy of
# an approximate posterior q is KL(p(x | z) ‖ q), averaged over M = 16
# measurement draws z = h(x*) + v, where x* is a random truth sample. The 95%
# interval comes from the spread across draws. The importance-sampling floor is
# about 1/√ESS, so a case counts as resolved only if its interval excludes zero
# and the smallest ESS is at least 100.
#
# Noise. Importance sampling from the prior needs the likelihood to be no
# narrower than a few percent of the prior's spread in z. Each σ is therefore
# 0.2 × the predicted linear standard deviation of that measurement at that
# epoch (for angles, of the wider of the two). This is informative but
# resolvable; it is not real deep-space-network accuracy (see the Gaps).
#
# (a) Where along the orbit does one Gaussian's posterior fail, and does a
#     criterion see it coming? P0's posterior KL is shown against the epoch,
#     beside two criteria: the exact state negentropy J, which is the same for
#     every measurement type, and P4's measurement-space criterion, which is not.
# (b) H5: posterior KL against the number of components at the approach to
#     perilune (0.45 T) and at perilune (0.5 T), with range measurements. P3 and
#     P4 split greedily to a cap K; P2's threshold is swept.
#
#   julia --project=scripts -t auto scripts/e5_posterior.jl

@isdefined(FIGDIR) || include(joinpath(@__DIR__, "common.jl"))

const E5_SEED = 505
const E5_FRACS = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0]
const E5_CAPS = [1, 3, 5, 9, 15, 21, 31]
const E5_TAUS = 10.0 .^ (-7:0)
const E5_N = 500_000
const E5_M = 16
const E5_RATIO = 0.2

function scaled_measurements(origin, xlin, Plin, ratio)
    std1(m) = sqrt.(diag(Matrix(measurement_jacobian(m, xlin)) * Plin * Matrix(measurement_jacobian(m, xlin))'))
    return [("range", Range(origin, ratio * std1(Range(origin, 1.0))[1])),
            ("range-rate", RangeRate(origin, ratio * std1(RangeRate(origin, 1.0))[1])),
            ("angles", Angles(origin, ratio * maximum(std1(Angles(origin, 1.0)))))]
end

function expected_posterior_kl(q::GaussianMixture, meas, Xt, H, W, x̄, lp0, idx, rng)
    kls = Float64[]; esss = Float64[]
    for j in idx
        z = measure(meas, view(Xt, :, j)) + sqrt(Matrix(measurement_cov(meas))) * randn(rng, length(measure(meas, view(Xt, :, j))))
        post = measurement_update(q, meas, z)
        pw = GaussianMixture(post.w, [W * (m - x̄) for m in post.m], [W * P * W' for P in post.P])
        kl, ess = posterior_kl(lp0, measurement_loglik(meas, z, Xt), mixture_logpdf(pw, H))
        push!(kls, kl); push!(esss, ess)
    end
    m = mean(kls); se = std(kls) / sqrt(length(kls))
    lo, hi = m - 2.13se, m + 2.13se                 # t(15), 95%
    ess = minimum(esss)
    return (; value = m, lo, hi, ess, resolved = lo > 0 && ess >= 100)
end

# smallest component count at which a front reaches KL ≤ ε (interval upper bound)
mink(front, ε) = (k = findfirst(q -> q[2].hi <= ε, sort(front; by = first)); k === nothing ? "–" : string(sort(front; by = first)[k][1]))

function e5_run()
    set_theme_paper!()
    t_start = time()
    nrho = t2_nrho()
    sys = nrho.model.system
    T = nrho.period
    P0 = scaled_prior(1.0, 10 / sys.l_star, 1e-4 * sys.t_star / sys.l_star)
    S0 = Matrix(cholesky(Symmetric(Matrix(P0))).L)
    origin = SVector(-sys.μ, 0.0, 0.0)
    ts = E5_FRACS .* T
    X0 = sample_gaussian(Xoshiro(E5_SEED), collect(nrho.x0), P0, E5_N)
    Ξ = S0 \ (X0 .- nrho.x0)
    lp0 = gaussian_logpdf(Ξ, zeros(6), Matrix(1.0I, 6, 6))
    clouds = propagate_samples(nrho.model, X0, ts)
    lin = propagate_tensors(nrho.model, nrho.x0, last(ts); saveat = ts)
    Jexact = negentropy_series_gh(nrho, P0, ts)
    idx = rand(Xoshiro(E5_SEED + 1), 1:E5_N, E5_M)
    names = ["range", "range-rate", "angles"]

    # ---------------- (a) along the orbit ----------------
    rowsA = ["| epoch [T] | measurement | σ | exact state J | measurement criterion | P0 posterior KL [95% CI] | min ESS | resolved |",
             "|---|---|---|---|---|---|---|---|"]
    A_res = Dict(n => NamedTuple[] for n in names)
    for (k, t) in enumerate(ts)
        Φ, Ψ, x̄ = lin.Φ[k], lin.Ψ[k], lin.x[k]
        Plin = Φ * P0 * Φ'
        W = S0 \ symplectic_inverse(nrho.model, Φ)
        H = W * (clouds[k] .- x̄)
        for (name, meas) in scaled_measurements(origin, x̄, Plin, E5_RATIO)
            A, Q, _ = measurement_tensors(meas, Φ, Ψ, x̄, S0)
            crit = measurement_negentropy(A, Q)
            e = expected_posterior_kl(GaussianMixture(x̄, Plin), meas, clouds[k], H, W, x̄, lp0, idx, Xoshiro(E5_SEED + 10k))
            push!(A_res[name], (; t = E5_FRACS[k], crit, e))
            push!(rowsA, @sprintf("| %.2f | %s | %.3g | %.3g | %.3g | %.3g [%.3g, %.3g] | %.0f | %s |", E5_FRACS[k], name,
                                  sqrt(Matrix(measurement_cov(meas))[1, 1]), Jexact[k], crit, e.value, e.lo, e.hi, e.ess,
                                  e.resolved ? "yes" : "no"))
        end
    end
    figA = Figure(size = (820, 380))
    ax1 = Axis(figA[1, 1]; yscale = log10, xlabel = "measurement epoch [revolutions from apolune]",
               ylabel = "P0 posterior KL [nats]", title = "(a) one Gaussian, one update (hollow = not resolved)",
               yticks = decade_ticks(-4, 6), xticks = 0:0.1:1)
    ax2 = Axis(figA[1, 2]; yscale = log10, xlabel = "measurement epoch [revolutions from apolune]",
               ylabel = "criterion [nats]", title = "(b) state J (black) and measurement criterion",
               yticks = decade_ticks(-12, 6; step = 2), xticks = 0:0.1:1)
    markers = [:circle, :rect, :utriangle]
    for (i, n) in enumerate(names)
        r = A_res[n]
        x = [q.t for q in r]
        v = [max(q.e.value, 1e-4) for q in r]
        lines!(ax1, x, v; color = CAT[i], linestyle = STYLES[i], label = n)
        res = [q.e.resolved for q in r]
        scatter!(ax1, x[res], v[res]; color = CAT[i], marker = markers[i], markersize = 9)
        scatter!(ax1, x[.!res], v[.!res]; color = :white, strokecolor = CAT[i], strokewidth = 1.5, marker = markers[i], markersize = 9)
        lines!(ax2, x, max.([q.crit for q in r], 1e-12); color = CAT[i], linestyle = STYLES[i], label = n)
        scatter!(ax2, x, max.([q.crit for q in r], 1e-12); color = CAT[i], marker = markers[i], markersize = 7)
    end
    lines!(ax2, E5_FRACS, max.(Jexact, 1e-12); color = INK, linewidth = 2, label = "state J")
    axislegend(ax1; position = :lt, labelsize = 9)
    axislegend(ax2; position = :lt, labelsize = 9)
    fa = save_figure(figA, "e5_along_orbit")

    # ---------------- (b) H5 at equal component count ----------------
    rowsB = ["| epoch | policy | posterior KL [95% CI] (components) |", "|---|---|---|"]
    rowsK = ["| epoch | target: fraction of P0's posterior KL | target [nats] | P2 needs K | P3 needs K | P4 needs K |", "|---|---|---|---|---|---|"]
    figB = Figure(size = (820, 400))
    for (col, frac) in enumerate((0.45, 0.5))
        k = findfirst(==(frac), E5_FRACS)
        t = ts[k]
        Φ, x̄ = lin.Φ[k], lin.x[k]
        Plin = Φ * P0 * Φ'
        W = S0 \ symplectic_inverse(nrho.model, Φ)
        H = W * (clouds[k] .- x̄)
        meas = Dict(scaled_measurements(origin, x̄, Plin, E5_RATIO))["range"]
        prior = GaussianMixture(nrho.x0, P0)
        post(pol) = (q = propagate_policy(pol, nrho.model, prior, [t])[1];
                     (ncomponents(q), expected_posterior_kl(q, meas, clouds[k], H, W, x̄, lp0, idx, Xoshiro(E5_SEED + 99))))
        p2 = [post(EntropyTrigger(τ; dt = T / 32, max_components = 31)) for τ in E5_TAUS]
        p3 = [post(HorizonSplit(0.0; max_components = K)) for K in E5_CAPS]
        p4 = [post(HorizonSplit(0.0; measurement = meas, max_components = K)) for K in E5_CAPS]
        ax = Axis(figB[1, col]; xscale = log10, yscale = log10, xticks = (E5_CAPS, string.(E5_CAPS)),
                  xlabel = "mixture components K", ylabel = col == 1 ? "posterior KL after one update [nats]" : "",
                  title = @sprintf("range measurement at %.2f T", frac), yticks = decade_ticks(-4, 6))
        scatter!(ax, first.(p2), max.([e.value for (_, e) in p2], 1e-4); color = CAT[1], marker = :diamond, markersize = 8,
                 label = "P2 entropy trigger (τ sweep)")
        for (i, (v, lab)) in enumerate(((p3, "P3 horizon, state"), (p4, "P4 horizon, measurement")))
            lines!(ax, first.(v), max.([e.value for (_, e) in v], 1e-4); color = CAT[i+1], linestyle = STYLES[i+1], label = lab)
            scatter!(ax, first.(v), max.([e.value for (_, e) in v], 1e-4); color = CAT[i+1], markersize = 7)
        end
        col == 1 && Legend(figB[2, 1:2], ax; orientation = :horizontal, labelsize = 9, tellheight = true)
        fmt(v) = join([@sprintf("K = %d: %.3g [%.3g, %.3g]", K, e.value, e.lo, e.hi) for (K, e) in v], "; ")
        push!(rowsB, @sprintf("| %.2f T | P2 entropy trigger | %s |", frac,
                              join([@sprintf("τ = %.0e: %.3g [%.3g, %.3g] (K = %d)", τ, e.value, e.lo, e.hi, K) for (τ, (K, e)) in zip(E5_TAUS, p2)], "; ")))
        push!(rowsB, @sprintf("| %.2f T | P3 horizon, state | %s |", frac, fmt(p3)))
        push!(rowsB, @sprintf("| %.2f T | P4 horizon, measurement | %s |", frac, fmt(p4)))
        push!(rowsB, @sprintf("| %.2f T | smallest ESS over all runs | %.0f |", frac, minimum(e.ess for (_, e) in vcat(p2, p3, p4))))
        p0kl = p3[1][2].value                       # K = 1 is P0
        for f in (0.5, 0.25, 0.1)
            ε = f * p0kl
            push!(rowsK, @sprintf("| %.2f T | %g | %.3g | %s | %s | %s |", frac, f, ε, mink(p2, ε), mink(p3, ε), mink(p4, ε)))
        end
    end
    fb = save_figure(figB, "e5_policies")

    # the split directions of P3 and P4 at perilune
    sol = propagate_tensors(nrho.model, nrho.x0, 0.5T)
    B = whitened_tensor(sol.Φ[end], sol.Ψ[end], S0)
    λ = sort(eigvals(Symmetric(sum(B[a, :, :] * B[a, :, :] for a in 1:6))); rev = true)
    dirs = String[]
    for (n, meas) in scaled_measurements(origin, sol.x[end], sol.Φ[end] * P0 * sol.Φ[end]', E5_RATIO)
        _, Q, _ = measurement_tensors(meas, sol.Φ[end], sol.Ψ[end], sol.x[end], S0)
        push!(dirs, @sprintf("%s %.5f", n, abs(dot(first(max_nonlinearity_direction(B)), first(max_nonlinearity_direction(Q))))))
    end

    body = """
    9:2 NRHO, prior 1 × (10 km, 10 cm/s) at apolune. One measurement from the Earth's centre, with
    σ = $(E5_RATIO) × its predicted linear standard deviation (for angles, of the wider axis). The
    exact posterior comes from N = 5×10⁵ importance-weighted prior samples. KL is averaged over
    $(E5_M) measurement draws, with a 95% interval from their spread. A case is resolved if its
    interval excludes zero and its smallest ESS is at least 100.

    **(a) Along the orbit.**

    $(join(rowsA, "\n"))

    ![E5 along the orbit]($fa)

    *(a) Posterior KL of a single linear Gaussian (P0) after one EKF update, against the
    measurement epoch, for each measurement type. Hollow markers are not resolved (interval
    includes zero, or ESS < 100). The error is resolved only between 0.45 and 0.6 T, around
    perilune. (b) The two criteria at the same epochs: the exact state negentropy J (black), which
    is the same for all three measurement types, and the measurement-space criterion (coloured),
    which differs by type. The state J peaks where the posterior error does. The measurement
    criterion does not order the measurement types the way their posterior errors are ordered.*

    **(b) H5: posterior accuracy against component count**, range measurements.

    $(join(rowsB, "\n"))

    Smallest number of components whose posterior KL is below a fraction of P0's, counting a
    target as reached when the upper end of the 95% interval is below it ("–" = not reached with
    31 components):

    $(join(rowsK, "\n"))

    **Split directions at perilune.** The whitened nonlinearity is essentially rank one: the
    eigenvalues of Σₐ Bₐ², normalized, are $(join([@sprintf("%.1e", x / λ[1]) for x in λ], ", ")). So the state
    and measurement criteria choose almost the same split direction, |u_state · u_meas| =
    $(join(dirs, "; ")). At equal K, P3 and P4 then split almost identically. They can only
    differ in how many splits they ask for at a given threshold.

    ![E5 policies]($fb)

    *Posterior KL after one Gaussian-sum update against the number of prior components, for range
    measurements at the approach to perilune and at perilune. P3 and P4 split greedily to K
    components; P2 is shown across its threshold sweep.*
    """
    record!("E5", "posterior accuracy after one update", body; seed = E5_SEED, runtime = time() - t_start)
end

e5_run()
