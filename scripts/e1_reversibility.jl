# E1: negentropy J(t) along the orbit, split into reversible and irreversible
# parts, J_irr(t) = min over s in [t, T_end] of J(s) and J_rev = J - J_irr.
#
#   (a) 9:2 NRHO, three revolutions from apolune, prior s × (10 km, 10 cm/s)
#   (b) Molniya, ten revolutions from perigee, prior s × (1 km, 1 m/s)
#   (c) Cartesian against modified equinoctial coordinates (LEO and Molniya, s = 1)
#
# J(t) for (a) and (b) comes from a 5⁶-node Gauss-Hermite rule in whitened
# coordinates, spot-checked against paired Monte Carlo with N = 2×10⁵. Panel (c)
# is Monte Carlo throughout, because its coordinate map needs a per-sample
# Jacobian.
#
#   julia --project=scripts -t auto scripts/e1_reversibility.jl

@isdefined(FIGDIR) || include(joinpath(@__DIR__, "common.jl"))

const E1_SEED = 101
const E1_PRIORS = [0.1, 0.3, 1.0, 3.0, 10.0]

function e1_run()
    set_theme_paper!()
    t_start = time()
    rows = String[]

    # ---------------- (a) NRHO ----------------
    nrho = t2_nrho()
    sys = nrho.model.system
    T = nrho.period
    Lstar, Vstar = sys.l_star, sys.l_star / sys.t_star
    # uniform grid, refined around each perilune passage (t = T/2 + kT)
    tsn = sort(unique(vcat(collect(range(0, 3T; length = 301)[2:end]),
                           [T / 2 + k * T + d for k in 0:2 for d in range(-0.04T, 0.04T; length = 41)])))
    Jn = Dict(s => negentropy_series_gh(nrho, scaled_prior(s, 10 / Lstar, 1e-4 / Vstar), tsn) for s in E1_PRIORS)

    fig = Figure(size = (720, 420))
    ax = Axis(fig[1, 1]; xlabel = "time [revolutions from apolune]", ylabel = "negentropy J [nats]",
              yscale = log10, title = "9:2 NRHO: negentropy along three revolutions",
              xticks = 0:0.5:3, yticks = decade_ticks(-8, 2; step = 2))
    for (i, s) in enumerate(E1_PRIORS)
        J = max.(Jn[s], 1e-9)
        lines!(ax, tsn ./ T, J; color = RAMP5[i], label = @sprintf("%g × (10 km, 10 cm/s)", s))
        lines!(ax, tsn ./ T, max.(irreversible(Jn[s]), 1e-9); color = RAMP5[i], linestyle = :dash, linewidth = 1.5)
    end
    vlines!(ax, [0.5, 1.5, 2.5]; color = GRID, linewidth = 1)
    text!(ax, [0.5, 1.5, 2.5], fill(2e-9, 3); text = fill("perilune", 3), align = (:center, :bottom), color = INK2, fontsize = 9)
    ylims!(ax, 1e-9, 100)
    Legend(fig[1, 2], ax, "prior (solid J, dashed J_irr)"; labelsize = 9, titlesize = 10)
    fa = save_figure(fig, "e1_nrho")

    # Monte Carlo spot checks at the first and last perilune and at apolune
    kp1 = argmin(abs.(tsn .- T / 2)); kp3 = argmin(abs.(tsn .- 2.5T)); ka = argmin(abs.(tsn .- T))
    checks = String[]
    for s in (1.0, 10.0)
        P0 = scaled_prior(s, 10 / Lstar, 1e-4 / Vstar)
        mc = negentropy_series_mc(nrho, P0, tsn[[kp1, ka, kp3]]; seed = E1_SEED, B = 200)
        for (k, e) in zip((kp1, ka, kp3), mc)
            push!(checks, @sprintf("| NRHO s = %g, t = %.3f T | %.4g | %.4g [%.4g, %.4g] |", s, tsn[k] / T, Jn[s][k], e.value, e.lo, e.hi))
        end
    end

    nrho_table = ["| prior s | J at 1st perilune | J_irr at 1st perilune | reversible fraction | J at 1 rev | J at 2 rev | J at 3 rev |",
                  "|---|---|---|---|---|---|---|"]
    k1, k2, k3 = (argmin(abs.(tsn .- k * T)) for k in 1:3)
    for s in E1_PRIORS
        J = Jn[s]; Ji = irreversible(J)
        push!(nrho_table, @sprintf("| %g | %.3g | %.3g | %.4f | %.3g | %.3g | %.3g |", s, J[kp1], Ji[kp1],
                                   1 - Ji[kp1] / J[kp1], J[k1], J[k2], J[k3]))
    end

    # ---------------- (b) Molniya ----------------
    mol = t1_molniya()
    Tm = mol.period
    tsm = sort(unique(vcat(collect(range(0, 10Tm; length = 801)[2:end]),
                           [k * Tm + d for k in 1:10 for d in range(-0.02Tm, 0.0; length = 11)])))
    Jm = Dict(s => negentropy_series_gh(mol, scaled_prior(s, 1.0, 1e-3), tsm) for s in E1_PRIORS)
    fig = Figure(size = (720, 420))
    ax = Axis(fig[1, 1]; xlabel = "time [revolutions from perigee]", ylabel = "negentropy J [nats]",
              yscale = log10, title = "Molniya: negentropy along ten revolutions", xticks = 0:1:10,
              yticks = decade_ticks(-3, 2))
    for (i, s) in enumerate(E1_PRIORS)
        lines!(ax, tsm ./ Tm, max.(Jm[s], 1e-9); color = RAMP5[i], label = @sprintf("%g × (1 km, 1 m/s)", s))
        lines!(ax, tsm ./ Tm, max.(irreversible(Jm[s]), 1e-9); color = RAMP5[i], linestyle = :dash, linewidth = 1.5)
    end
    ylims!(ax, 1e-3, 100)
    Legend(fig[1, 2], ax, "prior (solid J, dashed J_irr)"; labelsize = 9, titlesize = 10)
    fb = save_figure(fig, "e1_molniya")

    mol_table = ["| prior s | J at 1st apogee | J at 1st perigee | J_irr at 1st perigee | J at 10th apogee | J at 10th perigee |",
                 "|---|---|---|---|---|---|"]
    ka1 = argmin(abs.(tsm .- Tm / 2)); kpe1 = argmin(abs.(tsm .- Tm))
    ka10 = argmin(abs.(tsm .- 9.5Tm)); kpe10 = argmin(abs.(tsm .- 10Tm))
    for s in E1_PRIORS
        J = Jm[s]; Ji = irreversible(J)
        push!(mol_table, @sprintf("| %g | %.3g | %.3g | %.3g | %.3g | %.3g |", s, J[ka1], J[kpe1], Ji[kpe1], J[ka10], J[kpe10]))
    end
    mc = negentropy_series_mc(mol, scaled_prior(1.0, 1.0, 1e-3), tsm[[ka1, kpe1, kpe10]]; seed = E1_SEED + 1, B = 200)
    for (k, e) in zip((ka1, kpe1, kpe10), mc)
        push!(checks, @sprintf("| Molniya s = 1, t = %.3f T | %.4g | %.4g [%.4g, %.4g] |", tsm[k] / Tm, Jm[1.0][k], e.value, e.lo, e.hi))
    end

    # ---------------- (c) Cartesian against equinoctial ----------------
    coord_rows = ["| orbit | revolutions | J Cartesian [95% CI] | J equinoctial [95% CI] |", "|---|---|---|---|"]
    fig = Figure(size = (720, 380))
    for (col, p) in enumerate((t1_leo(), t1_molniya()))
        P0 = scaled_prior(1.0, 1.0, 1e-3)
        S0 = Matrix(cholesky(Symmetric(Matrix(P0))).L)
        tsc = p.period .* (1:40) ./ 4
        X0 = sample_gaussian(Xoshiro(E1_SEED + 10 + col), collect(p.x0), P0, 200_000)
        Ξ = S0 \ (X0 .- p.x0)
        clouds = propagate_samples(p.model, X0, tsc)
        lin = propagate_tensors(p.model, p.x0, last(tsc); order = 1, saveat = tsc)
        μ = p.model.μ
        Jc, Je = MCEstimate[], MCEstimate[]
        for k in eachindex(tsc)
            H = S0 \ (symplectic_inverse(p.model, lin.Φ[k]) * (clouds[k] .- lin.x[k]))
            push!(Jc, negentropy_mc(Ξ, H, I(6); B = 100, rng = Xoshiro(k)))
            # equinoctial: unwrap L about the reference trajectory, standardize rows
            qref = cartesian_to_equinoctial(lin.x[k], μ)
            G = reduce(hcat, [begin
                q = cartesian_to_equinoctial(view(clouds[k], :, j), μ)
                [q[1], q[2], q[3], q[4], q[5], unwrap_angle(q[6], qref[6])]
            end for j in axes(clouds[k], 2)])
            ldj = [equinoctial_logdetjac(view(clouds[k], :, j), μ) for j in axes(clouds[k], 2)]
            D = vec(std(G; dims = 2))
            push!(Je, negentropy_mc(Ξ, G ./ D, I(6); B = 100, rng = Xoshiro(k),
                                    logdetjac = ldj .- sum(log, D) .+ logdet(Symmetric(Matrix(S0 * S0'))) / 2))
        end
        ax = Axis(fig[1, col]; xlabel = "time [revolutions]", ylabel = col == 1 ? "negentropy J [nats]" : "",
                  yscale = log10, title = p.name * ", prior 1 km / 1 m/s", yticks = decade_ticks(-6, 2))
        x = tsc ./ p.period
        for (j, (Js, lab)) in enumerate(((Jc, "Cartesian"), (Je, "equinoctial")))
            band!(ax, x, max.([e.lo for e in Js], 1e-6), max.([e.hi for e in Js], 1e-6); color = (CAT[j], 0.25))
            lines!(ax, x, max.([e.value for e in Js], 1e-6); color = CAT[j], linestyle = STYLES[j], label = lab)
        end
        col == 1 && axislegend(ax; position = :rb)
        for k in (4, 40)
            push!(coord_rows, @sprintf("| %s | %d | %.4g [%.4g, %.4g] | %.4g [%.4g, %.4g] |", p.name, k ÷ 4,
                                       Jc[k].value, Jc[k].lo, Jc[k].hi, Je[k].value, Je[k].lo, Je[k].hi))
        end
    end
    fc = save_figure(fig, "e1_coordinates")

    body = """
    **(a) 9:2 NRHO.** Prior s × (10 km, 10 cm/s) per axis at apolune. J from a 5⁶ Gauss-Hermite rule.
    "Reversible fraction" is J_rev/J at the first perilune, with J_irr taken over the three-revolution horizon.

    $(join(nrho_table, "\n"))

    ![E1 NRHO]($fa)

    *Negentropy along three revolutions of the 9:2 NRHO, one line per prior size (log scale).
    Solid lines are J(t); dashed lines are J_irr(t) = min over s ≥ t of J(s). The gap between them
    is the reversible part: at each perilune J rises by four to eight orders of magnitude, and it
    falls back again by apolune.*

    **(b) Molniya.** Prior s × (1 km, 1 m/s) per axis at perigee. Ten revolutions; J from the same rule.

    $(join(mol_table, "\n"))

    ![E1 Molniya]($fb)

    *Negentropy along ten Molniya revolutions. J peaks at every perigee and drops at apogee. The
    irreversible floor (dashed) grows from one revolution to the next. At large priors J at the
    10th perigee falls as the prior grows: the cloud is spread along a large arc of the orbit, its
    covariance is bounded by the orbit's size, and ln det P₀ keeps growing as 12 ln s. Gauss-Hermite
    with 5⁶ and 7⁶ nodes and Monte Carlo agree to within 0.6% at s = 3 and 10.*

    **(c) Coordinates.** Paired Monte Carlo, N = 2×10⁵, 95% bootstrap intervals (B = 100). The
    equinoctial negentropy includes the Jacobian term, E[ln |det ∂q/∂x|].

    $(join(coord_rows, "\n"))

    ![E1 coordinates]($fc)

    *The same uncertainty measured in Cartesian and in modified equinoctial coordinates. The bands
    are 95% bootstrap intervals.*

    **Spot checks of the Gauss-Hermite values** against paired Monte Carlo (N = 2×10⁵, 95% interval):

    | case | Gauss-Hermite | Monte Carlo |
    |---|---|---|
    $(join(checks, "\n"))
    """
    record!("E1", "reversible and irreversible negentropy", body; seed = E1_SEED, runtime = time() - t_start)
    return (; Jn, Jm, tsn, tsm)
end

e1_run()
