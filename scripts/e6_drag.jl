# E6: the entropy ledger with drag. On a non-conservative flow, I2 needs the
# divergence term:
#
#   H[p_t] = H[p₀] + ∫₀ᵗ E[∇·f] ds,
#   J(t) = ½ ln det Σ_t - ½ ln det P₀ - ∫₀ᵗ E[∇·f] ds.
#
# T3: circular LEO at 400 km with drag in an exponential atmosphere, prior
# 1 km / 1 m/s, one day. The ledger is checked against an independent route:
# each sample's own ln det Φ (from its STM) gives E[ln det Dφ], which equals
# ∫E[∇·f] by Liouville's formula. The divergence term is then compared with J
# and with the error made by leaving it out ("naive I2"). Drag is scaled ×1,
# ×10 and ×100 (β = 0.011, 0.11 and 1.1 m²/kg): ×10 is roughly a high-solar-
# activity density, and ×100 an extreme area-to-mass ratio.
#
#   julia --project=scripts -t auto scripts/e6_drag.jl

@isdefined(FIGDIR) || include(joinpath(@__DIR__, "common.jl"))

const E6_SEED = 606
const E6_SCALES = [1.0, 10.0, 100.0]

function e6_case(scale; N = 20_000)
    p = t3_leo_drag(; β = 0.011 * scale)
    m = p.model
    day = 86400.0
    ts = collect(range(0, day; length = 97)[2:end])            # every 15 minutes
    P0 = scaled_prior(1.0, 1.0, 1e-3)
    X0 = sample_gaussian(Xoshiro(E6_SEED), collect(p.x0), P0, N)
    # per-sample trajectory and ln det Φ at every output time
    ldΦ = zeros(length(ts), N)
    Xs = [zeros(6, N) for _ in ts]
    Threads.@threads for j in 1:N
        sol = propagate_tensors(m, view(X0, :, j), day; order = 1, saveat = ts)
        for k in eachindex(ts)
            ldΦ[k, j] = logdet(sol.Φ[k])
            Xs[k][:, j] = sol.x[k]
        end
    end
    # time integral of the sample mean of ∇·f (trapezoid; ∇·f evaluated at the samples)
    divmean = [mean(flow_divergence(m, view(Xs[k], :, j)) for j in 1:N) for k in eachindex(ts)]
    div0 = mean(flow_divergence(m, view(X0, :, j)) for j in 1:N)
    tt = vcat(0.0, ts)
    dd = vcat(div0, divmean)
    integral = cumsum((dd[1:end-1] .+ dd[2:end]) ./ 2 .* diff(tt))
    E_ld = vec(mean(ldΦ; dims = 2))
    D = collect(p.scales)
    Jtrue = [negentropy_mc(X0 ./ D, Xs[k] ./ D, I(6); B = 50, logdetjac = ldΦ[k, :], rng = Xoshiro(k)) for k in eachindex(ts)]
    Jnaive = [negentropy_mc(X0 ./ D, Xs[k] ./ D, I(6); B = 50, rng = Xoshiro(k)) for k in eachindex(ts)]
    return (; ts, E_ld, integral, Jtrue, Jnaive, β = m.β)
end

function e6_run()
    set_theme_paper!()
    t_start = time()
    res = [e6_case(s) for s in E6_SCALES]

    fig = Figure(size = (820, 380))
    ax1 = Axis(fig[1, 1]; xlabel = "time [h]", ylabel = "entropy lost to drag, -ΔH [nats]", yscale = log10,
               title = "(a) ledger: E[ln det Φ] (lines) against ∫E[∇·f] dt (markers)", yticks = decade_ticks(-7, 0))
    ax2 = Axis(fig[1, 2]; xlabel = "time [h]", ylabel = "J [nats]",
               title = "(b) negentropy with the divergence term and without it")
    for (i, r) in enumerate(res)
        h = r.ts ./ 3600
        lab = @sprintf("β = %.3g m²/kg", r.β)
        lines!(ax1, h, max.(-r.E_ld, 1e-9); color = RAMP5[2i - 1], label = lab)
        scatter!(ax1, h[4:8:end], max.(-r.integral[4:8:end], 1e-9); color = RAMP5[2i - 1], marker = :circle, markersize = 6)
        lines!(ax2, h, [e.value for e in r.Jtrue]; color = RAMP5[2i - 1], label = lab)
        lines!(ax2, h, [e.value for e in r.Jnaive]; color = RAMP5[2i - 1], linestyle = :dash)
    end
    axislegend(ax1; position = :rb, labelsize = 9)
    text!(ax2, 0.02, 0.98; text = "solid: with the term; dashed: without", space = :relative,
          align = (:left, :top), fontsize = 9, color = INK2)
    f = save_figure(fig, "e6_drag")

    rows = ["| β [m²/kg] | -E[ln det Φ] after 1 day [nats] | -∫E[∇·f] dt [nats] | relative closure error | J with the term [95% CI] | J without it (naive I2) | relative error of naive I2 |",
            "|---|---|---|---|---|---|---|"]
    for r in res
        a, b = -r.E_ld[end], -r.integral[end]
        Jt, Jn = r.Jtrue[end], r.Jnaive[end]
        push!(rows, @sprintf("| %.3g | %.4g | %.4g | %.1e | %.4f [%.4f, %.4f] | %.4f | %.1e |", r.β, a, b, abs(a - b) / a,
                             Jt.value, Jt.lo, Jt.hi, Jn.value, abs(Jn.value - Jt.value) / Jt.value))
    end
    body = """
    T3: circular LEO at 400 km with exponential-atmosphere drag, prior 1 km / 1 m/s, one day,
    N = 2×10⁴ samples. Drag makes the flow contract phase volume (∇·f = -4·10³ β ρ |v| < 0), so the
    entropy of the propagated density falls. "Relative closure error" compares two independent
    computations of that loss: the mean of each sample's own ln det Φ, and the time integral of the
    mean divergence (trapezoid, 15-minute steps).

    $(join(rows, "\n"))

    ![E6 drag]($f)

    *(a) Entropy lost to drag over one day, computed two ways. Lines are the mean ln det Φ from each
    sample's state transition matrix; markers are the integral of the mean flow divergence. (b) J
    with the divergence term (solid) and without it (dashed). At realistic drag the term is many
    orders of magnitude smaller than J. The ledger closes, but leaving the term out changes nothing
    that matters here.*
    """
    record!("E6", "entropy ledger with drag", body; seed = E6_SEED, runtime = time() - t_start)
    return res
end

e6_run()
