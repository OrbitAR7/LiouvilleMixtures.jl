# E2: the refit ratchet. Covariance volume ½ ln det Σ(t)/det P₀ of a single
# Gaussian refitted on a schedule (P1), against the exact pushforward, whose
# volume is J(t) by I2. 9:2 NRHO, three revolutions from apolune, prior
# 1 × (10 km, 10 cm/s).
#
#   H1: with exact refits the volume never decreases from one refit to the next,
#       while the exact volume rises and falls.
#   H3: the volume gained depends mainly on where the refits fall, not on how
#       many there are.
#
# Exact refits use a 5⁶ Gauss-Hermite rule. The worst case is re-run with 7⁶ as
# a convergence check. Cubature refits (the unscented transform with α = 1,
# β = 0, κ = 0) are shown for comparison.
#
#   julia --project=scripts -t auto scripts/e2_ratchet.jl

@isdefined(FIGDIR) || include(joinpath(@__DIR__, "common.jl"))

const E2_SEED = 202

volume(gm, P0) = (logdet(Symmetric(Matrix(mixture_mean_cov(gm)[2]))) - logdet(Symmetric(Matrix(P0)))) / 2

function e2_run()
    set_theme_paper!()
    t_start = time()
    nrho = t2_nrho()
    sys = nrho.model.system
    T = nrho.period
    P0 = scaled_prior(1.0, 10 / sys.l_star, 1e-4 * sys.t_star / sys.l_star)
    prior = GaussianMixture(nrho.x0, P0)
    ts = sort(unique(vcat(collect(range(0, 3T; length = 361)[2:end]),
                          [T / 2 + k * T + d for k in 0:2 for d in range(-0.04T, 0.04T; length = 41)])))
    exact = negentropy_series_gh(nrho, P0, ts)

    cadences = [(T / 8, "every T/8"), (T / 4, "every T/4"), (T / 2, "every T/2"), (T, "every T")]
    phases = [(0.0, "at apolune"), (T / 4, "a quarter after apolune"), (T / 2, "at perilune"),
              (3T / 4, "a quarter before apolune")]
    moon = SVector(1 - sys.μ, 0.0, 0.0)
    R_moon = 1737.4 / sys.l_star
    # Run a schedule. If integration fails (a node of a refitted Gaussian falls
    # into the point-mass Moon), stop at the last refit that succeeded. Then
    # report the quadrature weight of that refit's nodes whose next-segment
    # trajectory passes below the lunar surface (failed nodes count as impacts).
    function run(pol)
        try
            return [volume(gm, P0) for gm in propagate_policy(pol, nrho.model, prior, ts)], Inf, 0.0
        catch err
            err isa CompositeException || err isa TaskFailedException || err isa ErrorException || rethrow()
        end
        ends = filter(t -> 0 < t <= 3T, [pol.phase + k * pol.Δ for k in 0:ceil(Int, 3T / pol.Δ)])
        last_ok = 0.0
        for e in ends
            ok = try propagate_policy(pol, nrho.model, prior, [e]); true catch; false end
            ok || break
            last_ok = e
        end
        keep = ts .<= last_ok
        v = fill(NaN, length(ts))
        v[keep] = [volume(gm, P0) for gm in propagate_policy(pol, nrho.model, prior, ts[keep])]
        g = propagate_policy(pol, nrho.model, prior, [last_ok])[1]
        Ξ, W = gauss_hermite_tensor(6, pol.n_quad)
        X = g.m[1] .+ Matrix(sqrt(Symmetric(Matrix(g.P[1])))) * Ξ
        grid = collect(range(0, pol.Δ; length = 400))
        hit = falses(size(X, 2))
        Threads.@threads for j in axes(X, 2)
            hit[j] = try
                minimum(norm(u[1:3] - moon) for u in propagate_trajectory(nrho.model, X[:, j], grid)) < R_moon
            catch
                true
            end
        end
        return v, last_ok, sum(W[hit])
    end
    # refit epochs of a schedule within (0, 3T]
    refit_times(Δ, φ) = filter(t -> 0 < t <= 3T, [φ + k * Δ for k in 0:ceil(Int, 3T / Δ)])

    # Monte Carlo probability that a draw from the refitted Gaussian g passes below
    # the lunar surface within the next interval h. The trajectory is sampled at
    # 4000 points per revolution. A failed integration (passing through the
    # point mass) counts as an impact. Wilson 95% interval.
    function impact_probability(g, h; n = 4000, seed = E2_SEED)
        X = sample_gaussian(Xoshiro(seed), collect(g.m[1]), Matrix(g.P[1]), n)
        grid = collect(range(0, h; length = max(400, ceil(Int, 4000 * h / T))))
        hit = falses(n)
        Threads.@threads for j in 1:n
            hit[j] = try
                minimum(norm(u[1:3] - moon) for u in propagate_trajectory(nrho.model, X[:, j], grid)) < R_moon
            catch
                true
            end
        end
        p = count(hit) / n
        z = 1.96
        c = (p + z^2 / (2n)) / (1 + z^2 / n)
        w = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / (1 + z^2 / n)
        return (; p, lo = max(c - w, 0.0), hi = min(c + w, 1.0))
    end
    # impact probability at every refit of a schedule up to `upto`; returns (max, epoch of max, first epoch above 1%)
    function impacts(pol, upto)
        r = filter(t -> t < min(upto, 3T) + 1e-12 && t < 3T, refit_times(pol.Δ, pol.phase))
        isempty(r) && return (nothing, NaN, NaN)
        gs = propagate_policy(pol, nrho.model, prior, r)
        ps = [impact_probability(g, min(pol.Δ, 3T - t); seed = E2_SEED + round(Int, 1000t)) for (g, t) in zip(gs, r)]
        k = argmax([q.p for q in ps])
        first_above = (i = findfirst(q -> q.p > 0.01, ps); i === nothing ? NaN : r[i])
        return (ps[k], r[k], first_above)
    end
    at(v, t) = v[argmin(abs.(ts .- t))]
    min_step(v, Δ, φ) = (r = refit_times(Δ, φ); minimum(diff(vcat(0.0, [at(v, t) for t in r]))))

    rows = ["| schedule | refit rule | refits in 3 rev | volume at 3T [nats] | gain over exact at 3T [nats] | smallest step between refits [nats] | largest lunar-impact probability of a refitted Gaussian [95% CI] (refit epoch) |",
            "|---|---|---|---|---|---|---|"]
    impact_notes = String[]
    res = Dict{Tuple{Float64,Float64,Symbol},Vector{Float64}}()
    breakdowns = String[]
    for (Δ, φ, lab) in vcat([(Δ, Δ, l) for (Δ, l) in cadences], [(T, φ == 0 ? T : φ, l) for (φ, l) in phases[2:end]])
        for method in (:quadrature, :unscented)
            pol = ScheduledRefit(Δ; phase = φ, method)
            v, last_ok, impact = run(pol)
            res[(Δ, φ, method)] = v
            rule = method === :quadrature ? "exact (GH 5⁶)" : "cubature"
            imp, t_imp, first_above = impacts(pol, isfinite(last_ok) ? last_ok : 3T)
            impcol = imp === nothing ? "–" : @sprintf("%.3f [%.3f, %.3f] (%.3f T)", imp.p, imp.lo, imp.hi, t_imp / T)
            flag = isnan(first_above) ? "" : " †"
            isnan(first_above) || push!(impact_notes, @sprintf("%s, %s: the Gaussian refitted at %.3f T is the first with more than 1%% lunar-impact probability; the largest is %.3f [%.3f, %.3f] at %.3f T.",
                                                               lab, rule, first_above / T, imp.p, imp.lo, imp.hi, t_imp / T))
            if isfinite(last_ok)
                good = .!isnan.(v)
                rt = filter(<=(last_ok), refit_times(Δ, φ))
                vr = [v[argmin(abs.(ts .- t))] for t in rt]
                push!(rows, @sprintf("| %s | %s | %d | breaks down after %.2f T | – | %.3g | %s |", lab, rule,
                                     length(rt), last_ok / T, minimum(diff(vcat(0.0, vr))), impcol))
                push!(breakdowns, @sprintf("%s, %s: integration fails in the interval after the refit at %.2f T.", lab, rule, last_ok / T))
            else
                push!(rows, @sprintf("| %s | %s | %d | %.4g%s | %.4g | %.3g | %s |", lab, rule,
                                     length(refit_times(Δ, φ)), v[end], flag, v[end] - exact[end], min_step(v, Δ, φ), impcol))
            end
        end
    end

    # convergence of the exact refit in a case that runs through (one refit per revolution, at perilune)
    worst, last7, impact7 = run(ScheduledRefit(T; phase = T / 2, method = :quadrature, n_quad = 7))
    v5 = res[(T, T / 2, :quadrature)]
    k1 = argmin(abs.(ts .- T / 2))
    conv = isfinite(last7) ?
        @sprintf("""**Convergence.** One refit per revolution at perilune, rerun with the 7⁶ rule: the first
        refit (t = 0.5 T) agrees with the 5⁶ rule (%.4f against %.4f nats). It then breaks down after
        t = %.2f T, because its outermost nodes reach further and fall into the Moon.
        So every volume after the first perilune refit is not converged, and the totals of 48 to 56
        nats in the table are indicative only. The first step, which equals J at perilune, is robust.""",
                 worst[k1], v5[k1], last7 / T) :
        @sprintf("**Convergence.** One refit per revolution at perilune, 7⁶ rule: volume at 3T = %.4g nats (5⁶ rule: %.4g).",
                 worst[end], v5[end])

    fig = Figure(size = (760, 680))
    x = ts ./ T
    floor_ = 1e-6
    pos(v) = max.(v, floor_)
    function stopmark!(ax, v, color)
        k = findlast(!isnan, v)
        k !== nothing && k < length(v) && scatter!(ax, [x[k]], [pos(v)[k]]; color, marker = :xcross, markersize = 12)
    end
    common = (yscale = log10, yticks = decade_ticks(-6, 2), xticks = 0:0.5:3,
              ylabel = "½ ln det Σ / det P₀ [nats]")
    ax1 = Axis(fig[1, 1]; common..., title = "(a) refit cadence, exact refits (× = propagation breaks down)")
    for (i, (Δ, l)) in enumerate(cadences)
        v = res[(Δ, Δ, :quadrature)]
        lines!(ax1, x, pos(v); color = RAMP5[i+1], label = l)
        stopmark!(ax1, v, RAMP5[i+1])
    end
    lines!(ax1, x, pos(exact); color = INK, linewidth = 1.2, label = "exact pushforward (= J)")
    ax2 = Axis(fig[2, 1]; common..., xlabel = "time [revolutions from apolune]",
               title = "(b) one exact refit per revolution, by location")
    for (i, (φ, l)) in enumerate(phases)
        v = res[(T, φ == 0 ? T : φ, :quadrature)]
        lines!(ax2, x, pos(v); color = vcat(CAT, "#eda100")[i], linestyle = (:solid, :dash, :dot, :dashdot)[i], label = l)
    end
    lines!(ax2, x, pos(exact); color = INK, linewidth = 1.2, label = "exact pushforward (= J)")
    for ax in (ax1, ax2)
        vlines!(ax, [0.5, 1.5, 2.5]; color = GRID, linewidth = 1)
        ylims!(ax, floor_, 300)
        axislegend(ax; position = :rb, labelsize = 9, bgcolor = (:white, 0.85), framevisible = false)
    end
    f = save_figure(fig, "e2_ratchet")

    body = """
    9:2 NRHO, prior 1 × (10 km, 10 cm/s) per axis at apolune, three revolutions. The "volume" is
    ½ ln det Σ(t)/det P₀. For the exact pushforward it equals J(t), by I2. "Gain over exact" is the
    refitted volume minus the exact volume at 3T. "Smallest step" is the smallest change in volume
    from one refit to the next: H1 predicts it is ≥ 0 for exact refits.

    $(join(rows, "\n"))

    $conv

    **Lunar impacts.** The impact column is a Monte Carlo probability: 4000 draws from each
    refitted Gaussian, flown over the next refit interval, counting those that pass below the
    lunar surface (1737.4 km). The true pushforward has no such mass. † marks a schedule whose
    later refits are not physical: some refitted Gaussian in it has more than 1% probability of
    passing through the Moon. Its final volume should be read as indicative only.

    $(isempty(impact_notes) ? "No refitted Gaussian exceeds 1%." : join("- " .* impact_notes, "\n"))

    **Breakdowns.** For some schedules, a refitted Gaussian's quadrature nodes fall into the
    point-mass Moon, and integration fails. Those exact-refit curves stop at the last refit that
    propagated:

    $(isempty(breakdowns) ? "None." : join("- " .* breakdowns, "\n"))

    ![E2 ratchet]($f)

    *Covariance volume of a single Gaussian refitted on a schedule (coloured lines), against the
    exact pushforward (black, thin), on a log scale. The exact volume rises at every perilune and
    returns to near zero at apolune. A refit keeps whatever volume it finds, so after a refit the
    volume can only step up. Any schedule with a refit at a perilune locks in that perilune's
    volume, while refits near apolune stay close to the exact curve. (a) Refit cadence; × marks
    where propagation breaks down. (b) One refit per revolution, placed at different points on
    the orbit.*
    """
    record!("E2", "refit ratchet against cadence and location", body; seed = E2_SEED, runtime = time() - t_start)
    return (; ts, exact, res)
end

e2_run()
