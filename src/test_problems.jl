# =============================================================================
# Test problems T1 and T2 of the brief (T0 and T3 arrive with M2 and E6)
# =============================================================================

"""
    TestProblem

A dynamics model, an initial mean state and a reference period. Fields:

- `name`
- `model`: a [`DynamicsModel`](@ref)
- `x0`: initial state, in the model's units
- `period`: one revolution, in the model's time unit
- `scales`: length-6 vector of characteristic sizes of each state component, in
  the model's units. Used to make tensor comparisons dimensionless.
"""
struct TestProblem{M<:DynamicsModel}
    name::String
    model::M
    x0::SVector{6,Float64}
    period::Float64
    scales::SVector{6,Float64}
end

function _kepler_problem(name, model, a, e, i, Ω, ω, ν)
    μ = model.μ
    s = keplerian_to_cartesian(KeplerianElements(a, e, i, Ω, ω, ν); μ)
    x0 = SVector{6}(s.r..., s.v...)
    v = sqrt(μ / a)
    return TestProblem(name, model, x0, 2π * sqrt(a^3 / μ), SVector(a, a, a, v, v, v))
end

"""
    t1_leo(model = TwoBody())

T1 LEO: a = 7000 km, e = 0.01, starting at perigee. The brief fixes only `a`
and `e`. The other elements are chosen here: i = 51.6°, Ω = 0, ω = 0, ν = 0.
`model` is `TwoBody()` or `J2Gravity()`. The period is the Keplerian one.
"""
t1_leo(model::Union{TwoBody,J2Gravity} = TwoBody()) =
    _kepler_problem("T1 LEO", model, 7000.0, 0.01, deg2rad(51.6), 0.0, 0.0, 0.0)

"""
    t1_molniya(model = TwoBody())

T1 Molniya: a = 26 554 km, e = 0.72, i = 63.4° (from the brief). Chosen here:
ω = 270° (apogee over the northern hemisphere), Ω = 0, starting at perigee
(ν = 0).
"""
t1_molniya(model::Union{TwoBody,J2Gravity} = TwoBody()) =
    _kepler_problem("T1 Molniya", model, 26554.0, 0.72, deg2rad(63.4), 0.0, deg2rad(270.0), 0.0)

"""
    nrho_9_2(system = EARTH_MOON) -> PeriodicOrbit

The southern L2 halo orbit that makes 9 revolutions in 2 synodic months, the
9:2 NRHO. It follows the recipe that OrbitalMechanics.jl tests against the
literature initial state:

1. Continue the southern L2 halo family until perilune falls below 2000 km.
2. Bracket the period (2/9)·29.530588853 d.
3. Solve for that period with `orbit_with_period`.

The state is at apolune, on the xz-plane crossing, in nondimensional units.
Takes several seconds.
"""
function nrho_9_2(system::CR3BPSystem = EARTH_MOON)
    target = (2 / 9) * 29.530588853 * 86400 / system.t_star
    r_stop = 2000.0 / system.l_star
    family = halo_family(system, :L2; branch = :southern, n = 400, ds_max = 2e-2,
                         stop = o -> secondary_distance_range(o)[1] < r_stop)
    k = findfirst(i -> (family[i].period - target) * (family[i+1].period - target) <= 0,
                  1:length(family)-1)
    k === nothing && error("nrho_9_2: the halo family does not bracket the 9:2 period")
    return orbit_with_period(system, family[k], family[k+1], target)
end

"""
    t2_nrho(system = EARTH_MOON)

T2: the 9:2 southern L2 NRHO of [`nrho_9_2`](@ref) under `CR3BP(system)`,
starting at apolune. Scales are 1 (the units are already nondimensional).
"""
function t2_nrho(system::CR3BPSystem = EARTH_MOON)
    o = nrho_9_2(system)
    return TestProblem("T2 NRHO 9:2", CR3BP(system), o.x0, o.period, SVector{6}(ones(6)))
end
