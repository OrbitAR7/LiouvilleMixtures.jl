# E3: the leading-order negentropy predictor J ≈ ¼ΣB² + ½ΣB_abcB_bac (I4)
# against the exact negentropy, over every orbit problem, time and prior size.
# Tests H4.
#
# For each problem and output time, the predictor is evaluated once at a base
# prior. It scales exactly as s² with the prior size s, because B is linear in
# S₀. The exact J comes from Gauss-Hermite rules in whitened coordinates. A
# point is kept only where the 5⁶ and 7⁶ rules agree to 1%; the 7⁶ value is
# used. That drops the points below the noise floor of full-state integration,
# which is about 1e-7 nats for metre-level priors on Molniya. One prior per
# problem is also measured with paired Monte Carlo (N = 2×10⁵, 95% interval) as
# an independent check of the Gauss-Hermite values.
#
#   julia --project=scripts -t auto scripts/e3_predictor.jl

@isdefined(FIGDIR) || include(joinpath(@__DIR__, "common.jl"))

const E3_SEED = 303

function e3_run()
    set_theme_paper!()
    t_start = time()
    nrho = t2_nrho()
    sys = nrho.model.system
    Ln, Vn = 10 / sys.l_star, 1e-4 * sys.t_star / sys.l_star
    cases = [
        (t1_leo(), "T1 LEO, two-body", (1:8) ./ 4, [0.003, 0.01, 0.03, 0.1, 0.3, 1.0], 1.0, 1e-3, 0.1),
        (t1_leo(J2Gravity()), "T1 LEO, J2", (1:8) ./ 4, [0.003, 0.01, 0.03, 0.1, 0.3, 1.0], 1.0, 1e-3, 0.1),
        (t1_molniya(), "T1 Molniya", (1:8) ./ 8, [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], 1.0, 1e-3, 0.01),
        (nrho, "T2 9:2 NRHO", (1:24) ./ 24, [0.1, 0.3, 1.0, 3.0, 10.0], Ln, Vn, 1.0),
    ]
    T0curve = (k² -> log1p(2k²) / 2 / k²)          # exact / predicted for the T0 shear, J_pred = k²

    fig = Figure(size = (820, 640))
    rows = ["| problem | points kept (of total) | J_pred where exact/pred first leaves [0.9, 1.1] | largest J_pred with all points within 10% below it | Monte Carlo against Gauss-Hermite (prior s; median relative difference where resolved) |",
            "|---|---|---|---|---|"]
    allpts = Tuple{Float64,Float64}[]
    per_problem = Dict{String,Vector{Tuple{Float64,Float64}}}()
    for (ci, (p, name, fracs, priors, σr, σv, s_mc)) in enumerate(cases)
        ts = collect(fracs .* p.period)
        Sb = Matrix(Diagonal([fill(σr, 3); fill(σv, 3)]))
        sol = propagate_tensors(p.model, p.x0, last(ts); saveat = ts)
        Jb = [negentropy_leading(whitened_tensor(sol.Φ[k], sol.Ψ[k], Sb)) for k in eachindex(ts)]
        ax = Axis(fig[(ci - 1) ÷ 2 + 1, (ci - 1) % 2 + 1]; xscale = log10, xticks = decade_ticks(-6, 4; step = 2),
                  xlabel = ci > 2 ? "predicted J [nats]" : "", ylabel = isodd(ci) ? "exact J / predicted J" : "",
                  title = name, limits = (1e-7, 1e5, 0, 1.25))
        hspan!(ax, 0.9, 1.1; color = (GRID, 0.6))
        xs = 10.0 .^ range(-7, 5; length = 200)
        lines!(ax, xs, T0curve.(xs); color = INK2, linestyle = :dash, linewidth = 1, label = "T0 shear (exact)")
        pts = Tuple{Float64,Float64}[]
        Jgh = Dict{Float64,Vector{Float64}}()
        ntotal = 0
        for (i, s) in enumerate(priors)
            P0 = scaled_prior(s, σr, σv)
            J5 = negentropy_series_gh(p, P0, ts; n = 5)
            Jx = negentropy_series_gh(p, P0, ts; n = 7)
            Jgh[s] = Jx
            Jp = s^2 .* Jb
            keep = (Jx .> 0) .& (abs.(J5 .- Jx) .< 0.01 .* abs.(Jx))
            ntotal += length(ts)
            scatter!(ax, Jp[keep], (Jx ./ Jp)[keep]; color = RAMP5[clamp(round(Int, 1 + 4 * (i - 1) / (length(priors) - 1)), 1, 5)],
                     markersize = 8, strokecolor = :white, strokewidth = 0.5)
            append!(pts, zip(Jp[keep], (Jx ./ Jp)[keep]))
        end
        sort!(pts)
        append!(allpts, pts)
        per_problem[name] = pts
        first_out = findfirst(q -> abs(q[2] - 1) > 0.1, pts)
        Jfail = first_out === nothing ? Inf : pts[first_out][1]
        Jok = first_out === nothing ? pts[end][1] : (first_out == 1 ? 0.0 : pts[first_out-1][1])
        # Monte Carlo at one prior
        mc = negentropy_series_mc(p, scaled_prior(s_mc, σr, σv), ts; seed = E3_SEED + ci, B = 100)
        Jp_mc = s_mc^2 .* Jb
        res = [(Jp_mc[k], e, Jgh[s_mc][k]) for (k, e) in enumerate(mc) if e.lo > 0 && (e.hi - e.lo) < 0.5 * e.value]
        if !isempty(res)
            errorbars!(ax, first.(res), [e.value / j for (j, e, _) in res], [(e.value - e.lo) / j for (j, e, _) in res],
                       [(e.hi - e.value) / j for (j, e, _) in res]; color = CAT[2], whiskerwidth = 4)
            scatter!(ax, first.(res), [e.value / j for (j, e, _) in res]; color = :white, strokecolor = CAT[2],
                     strokewidth = 1.5, marker = :diamond, markersize = 8)
        end
        mcdev = isempty(res) ? NaN : median([abs(e.value / g - 1) for (_, e, g) in res])
        push!(rows, @sprintf("| %s | %d (%d) | %.3g | %.3g | s = %g; %s |", name, length(pts), ntotal, Jfail, Jok, s_mc,
                             isempty(res) ? "not resolved" : @sprintf("%d resolved, median %.4f", length(res), mcdev)))
    end
    Legend(fig[3, 1:2], [LineElement(color = INK2, linestyle = :dash), MarkerElement(color = RAMP5[3], marker = :circle),
                         MarkerElement(color = :white, strokecolor = CAT[2], strokewidth = 1.5, marker = :diamond)],
           ["T0 shear, exact ratio", "Gauss-Hermite, one colour per prior (light = small)", "Monte Carlo, 95% interval"];
           orientation = :horizontal, tellheight = true)
    f = save_figure(fig, "e3_predictor")

    # pooled: the worst |ratio - 1| in bins of predicted J
    bins = [(1e-7, 1e-3), (1e-3, 1e-2), (1e-2, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 10.0), (10.0, Inf)]
    shear(j) = log1p(2j) / 2            # the T0 closure: J = ½ ln(1 + 2 J_pred)
    pooled = ["| predicted J [nats] | points | median exact/pred | worst exact/pred | worst exact / ½ln(1+2 pred) |",
              "|---|---|---|---|---|"]
    for (a, b) in bins
        sel = [q for q in allpts if a <= q[1] < b]
        isempty(sel) && continue
        r = [q[2] for q in sel]
        c = [q[2] * q[1] / shear(q[1]) for q in sel]
        push!(pooled, @sprintf("| %g – %g | %d | %.3f | %.3f | %.3f |", a, b, length(r), median(r), r[argmax(abs.(r .- 1))],
                               c[argmax(abs.(c .- 1))]))
    end
    shear_dev(name, jmax) = maximum(abs(q[2] * q[1] / shear(q[1]) - 1) for q in per_problem[name] if q[1] < jmax)

    body = """
    Predicted against exact negentropy over every problem, output time and prior size. Times:
    - T1: quarter revolutions up to 2 revolutions (LEO), eighth revolutions up to 1 revolution (Molniya).
    - T2: 1/24 revolution up to 1 revolution.

    Priors: s × (1 km, 1 m/s) for T1, s × (10 km, 10 cm/s) for T2. "Leaves [0.9, 1.1]" is the
    smallest predicted J at which a point falls outside ±10%, with points sorted by predicted J.

    $(join(rows, "\n"))

    All problems pooled:

    $(join(pooled, "\n"))

    Largest deviation of the exact J from the shear closure ½ ln(1 + 2 J_pred):

    | problem | over J_pred < 1 nat | over J_pred < 10 nats |
    |---|---|---|
    $(join([@sprintf("| %s | %.1f%% | %.1f%% |", c[2], 100 * shear_dev(c[2], 1.0), 100 * shear_dev(c[2], 10.0)) for c in cases], "\n"))

    ![E3 predictor]($f)

    *Exact negentropy divided by the leading-order prediction, against the prediction. One panel
    per problem, one colour per prior size. The grey band is ±10%, and the dashed line is the
    exact ratio for the T0 shear map, ln(1 + 2k²)/(2k²) at J_pred = k². In all four problems the
    predictor is within 10% up to a predicted J of about 0.1 nats. Beyond that the points follow
    the shear curve: the true negentropy behaves like ½ ln(1 + 2 J_pred), far past leading order.
    On LEO this holds to 0.1% up to 10 nats, where the nonlinearity is the curvature of the arc, a
    quadratic shear like T0's. On Molniya and the NRHO it holds to about 12% up to 1 nat and
    departs beyond. This is an empirical observation, not a derived result. Diamonds are paired
    Monte Carlo with 95% intervals.*
    """
    record!("E3", "negentropy predictor against the exact negentropy", body; seed = E3_SEED, runtime = time() - t_start)
end

e3_run()
