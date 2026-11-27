# =============================================================================
# Propagation policies P0-P5
# =============================================================================
#
#   P0  LinearPropagation()                      single Gaussian, linear (STM)
#   P1  ScheduledRefit(Δ; phase, method)         single Gaussian, refit every Δ
#   P2  EntropyTrigger(τ; dt)                    split when the accumulated
#                                                per-step entropy difference > τ
#   P3  HorizonSplit(τ)                          split at each decision epoch if
#                                                the predicted negentropy to the
#                                                next epoch > τ (state space)
#   P4  HorizonSplit(τ; measurement)             same, measurement space
#   P5  HorizonSplit(τ; measurement, τ_merge)    P4 plus Runnalls merge-back
#
# Mixture components in P2-P5 are propagated linearly. With every threshold
# infinite these policies therefore reduce exactly to P0, and they differ only
# in where they split and merge. P1 has a cadence, not a threshold: on linear
# dynamics it equals P0, and with Δ = ∞ it is a single unscented (or quadrature)
# transform over the whole horizon.
#
# All times are in the model's time unit, measured from the prior epoch (t = 0).

"""
    Policy

Abstract type for a propagation policy. See [`propagate_policy`](@ref).
"""
abstract type Policy end

"""
    LinearPropagation()

P0: every component is propagated by its state transition matrix,
`m ↦ φ(m)` and `P ↦ Φ P Φᵀ`. It never splits or merges.
"""
struct LinearPropagation <: Policy end

"""
    ScheduledRefit(Δ; phase = 0, method = :unscented, α = 1, β = 0, κ = 0, n_quad = 5)

P1: a single Gaussian that is refitted at the times `phase + kΔ` (k = 0, 1, …)
in `(0, T]`.

- Between refits the Gaussian of the last refit is carried as a point cloud.
  Its moments are reported at the output times.
- `method = :unscented` uses the scaled unscented transform (`α`, `β`, `κ`).
  The defaults give the third-degree cubature rule, which has positive weights.
- `method = :quadrature` uses an `n_quad`⁶ Gauss-Hermite rule. To the accuracy
  of the rule this is exact moment matching, and it is the clean test of the
  ratchet (identity I3).
"""
struct ScheduledRefit <: Policy
    Δ::Float64
    phase::Float64
    method::Symbol
    α::Float64
    β::Float64
    κ::Float64
    n_quad::Int
    function ScheduledRefit(Δ::Real; phase::Real = 0.0, method::Symbol = :unscented,
                            α::Real = 1.0, β::Real = 0.0, κ::Real = 0.0, n_quad::Integer = 5)
        Δ > 0 || throw(ArgumentError("Δ must be positive"))
        method in (:unscented, :quadrature) || throw(ArgumentError("method must be :unscented or :quadrature"))
        new(Δ, phase, method, α, β, κ, n_quad)
    end
end

"""
    EntropyTrigger(τ; dt, max_components = 50, shrink = 0.7)

P2: the instantaneous entropy trigger of DeMars, Bishop and Jah (2013).

- **Measure.** On every step of length `dt` each component accumulates the
  entropy difference between its cubature-rule prediction and its linearized
  prediction, `ΔH = ½ ln det P_σ - ln |det A|`. Here `A` is the
  central-difference linearization in the sigma directions.
- **Trigger.** When the running sum exceeds `τ`, the component is split before
  the step, and its three children redo the step. The worst component is split
  first, until every component is within `τ` or there are `max_components`.
- **Direction.** The split is along the sigma direction with the largest
  second difference, measured in units of the predicted covariance.
- The children's sums start at zero.

The components themselves are propagated linearly.
"""
struct EntropyTrigger <: Policy
    τ::Float64
    dt::Float64
    max_components::Int
    shrink::Float64
    function EntropyTrigger(τ::Real; dt::Real, max_components::Integer = 50, shrink::Real = 0.7)
        dt > 0 || throw(ArgumentError("dt must be positive"))
        new(τ, dt, max_components, shrink)
    end
end

"""
    HorizonSplit(τ; measurement = nothing, τ_merge = 0, max_components = 50, shrink = 0.7)

P3, P4 and P5: decide splits at each decision epoch (the prior epoch and every
measurement epoch), looking ahead along the composed second-order map to the
next epoch.

- **P3** (`measurement = nothing`): the criterion is the leading-order state
  negentropy at the next epoch, [`predicted_negentropy`](@ref), and the split
  direction maximizes `Σ_a (uᵀB_a u)²`.
- **P4** (`measurement` given): the criterion is the leading-order negentropy of
  the predicted measurement, [`measurement_negentropy`](@ref), and the direction
  maximizes the measurement-space bending `Σ_c (uᵀQ_c u)²`.
- **P5** (`τ_merge > 0`): before splitting, the cheapest pair of components is
  merged repeatedly while its Runnalls bound ([`merge_cost`](@ref)) is below
  `τ_merge`.

The worst component is split first, and its children are re-evaluated, until
every component is within `τ` or there are `max_components`. Components are
propagated linearly between epochs.
"""
struct HorizonSplit{M<:Union{Nothing,MeasurementModel}} <: Policy
    τ::Float64
    measurement::M
    τ_merge::Float64
    max_components::Int
    shrink::Float64
end
HorizonSplit(τ::Real; measurement = nothing, τ_merge::Real = 0.0, max_components::Integer = 50,
             shrink::Real = 0.7) = HorizonSplit(float(τ), measurement, float(τ_merge), Int(max_components), float(shrink))

# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

_sym(P) = SMatrix{6,6,Float64,36}((P + P') / 2)

function _linear_step(model::DynamicsModel, m, P, dt)
    dt == 0 && return SVector{6,Float64}(m), _sym(P)
    s = propagate_tensors(model, m, dt; order = 1)
    Φ = s.Φ[end]
    return s.x[end], _sym(Φ * P * Φ')
end

function _linear_mixture(model::DynamicsModel, gm::GaussianMixture, dt)
    steps = [_linear_step(model, gm.m[i], gm.P[i], dt) for i in eachindex(gm.w)]
    return GaussianMixture(gm.w, first.(steps), last.(steps))
end

# Propagate each column of X through the successive steps dts (cumulative), and
# return the cloud after each step.
function _flow_points(model::DynamicsModel, X::AbstractMatrix, dts::AbstractVector)
    out = [similar(X, Float64) for _ in dts]
    Threads.@threads for j in axes(X, 2)
        x = SVector{6,Float64}(view(X, :, j))
        for (k, h) in enumerate(dts)
            x = h == 0 ? x : propagate_state(model, x, h)
            out[k][:, j] = x
        end
    end
    return out
end

# Standardized nodes and weights of a deterministic rule: columns ξ, with the
# point x = m + S ξ for S the symmetric square root of P.
function _sigma_rule(α, β, κ)
    N = 6
    λ = α^2 * (N + κ) - N
    γ = sqrt(N + λ)
    Ξ = hcat(zeros(N), γ * Matrix(1.0I, N, N), -γ * Matrix(1.0I, N, N))
    wi = 1 / (2 * (N + λ))
    wm = vcat(λ / (N + λ), fill(wi, 2N))
    wc = vcat(λ / (N + λ) + (1 - α^2 + β), fill(wi, 2N))
    return Ξ, wm, wc, γ
end

function _rule(p::ScheduledRefit)
    if p.method === :unscented
        Ξ, wm, wc, _ = _sigma_rule(p.α, p.β, p.κ)
        return Ξ, wm, wc
    else
        Ξ, w = gauss_hermite_tensor(6, p.n_quad)
        return Ξ, w, w
    end
end

function _rule_moments(Y::AbstractMatrix, wm, wc)
    μ = Y * wm
    D = Y .- μ
    return SVector{6,Float64}(μ), _sym((D .* wc') * D')
end

_principal_sqrt(P) = Matrix(sqrt(Symmetric(Matrix(P))))

# -----------------------------------------------------------------------------
# propagate_policy
# -----------------------------------------------------------------------------

"""
    propagate_policy(policy, model, prior, times; epochs = Float64[]) -> Vector{GaussianMixture}

Propagate the mixture `prior` (or a single Gaussian `GaussianMixture(m, P)`)
under `model` with `policy`, and return the mixture at each of the sorted,
positive output `times`.

- `epochs` are the decision epochs (measurement times) in `(0, max(times))`.
  [`HorizonSplit`](@ref) uses them; the other policies ignore them.
- An output time equal to an epoch reports the mixture arriving at that epoch,
  before any split or merge there.
"""
function propagate_policy end

function _check_times(times)
    issorted(times) && all(>(0), times) || throw(ArgumentError("times must be sorted and positive"))
end

function propagate_policy(::LinearPropagation, model::DynamicsModel, prior::GaussianMixture,
                          times::AbstractVector; epochs = Float64[])
    _check_times(times)
    out = GaussianMixture[]
    cur, t = prior, 0.0
    for tk in times
        cur = _linear_mixture(model, cur, tk - t)
        t = tk
        push!(out, cur)
    end
    return out
end

function propagate_policy(p::ScheduledRefit, model::DynamicsModel, prior::GaussianMixture,
                          times::AbstractVector; epochs = Float64[])
    _check_times(times)
    ncomponents(prior) == 1 || throw(ArgumentError("ScheduledRefit propagates a single Gaussian"))
    T = last(times)
    refits = isfinite(p.Δ) ? [p.phase + k * p.Δ for k in 0:floor(Int, (T - p.phase) / p.Δ)] :
             [p.phase]
    refits = filter(r -> 0 < r <= T, refits)
    ends = unique(vcat(refits, T))
    Ξ, wm, wc = _rule(p)
    m, P = prior.m[1], prior.P[1]
    out = Vector{GaussianMixture}(undef, length(times))
    ts = 0.0
    for te in ends
        # outputs inside this segment, then the segment end
        idx = findall(t -> ts < t <= te, times)
        targets = unique(vcat(times[idx], te))
        X = m .+ _principal_sqrt(P) * Ξ
        clouds = propagate_samples(model, X, targets .- ts)     # one integration per node
        for (k, tk) in enumerate(targets)
            μk, Pk = _rule_moments(clouds[k], wm, wc)
            for i in idx
                times[i] == tk && (out[i] = GaussianMixture(μk, Pk))
            end
        end
        m, P = _rule_moments(clouds[end], wm, wc)           # refit
        ts = te
    end
    return out
end

# One step of the P2 measure for a component: returns (ΔH, s), where s is the
# one-sigma split direction (a column of the principal square root of P).
function _entropy_step(model::DynamicsModel, m, P, h)
    Ξ, wm, wc, γ = _sigma_rule(1.0, 0.0, 0.0)
    S = _principal_sqrt(P)
    Y = _flow_points(model, m .+ S * Ξ, [h])[1]
    μσ, Pσ = _rule_moments(Y, wm, wc)
    A = (Y[:, 2:7] - Y[:, 8:13]) / (2γ)
    ΔH = logdet(cholesky(Symmetric(Matrix(Pσ)))) / 2 - logabsdet(A)[1]
    C = cholesky(Symmetric(Matrix(Pσ)))
    b = [(Y[:, 1+i] + Y[:, 7+i]) / 2 - Y[:, 1] for i in 1:6]
    i_best = argmax([dot(bi, C \ bi) for bi in b])
    return ΔH, SVector{6,Float64}(S[:, i_best])
end

# whitened direction (Cholesky coordinates of P) of a one-sigma vector s
_whiten_direction(P, s) = cholesky(Symmetric(Matrix(P))).L \ s

function propagate_policy(p::EntropyTrigger, model::DynamicsModel, prior::GaussianMixture,
                          times::AbstractVector; epochs = Float64[])
    _check_times(times)
    T = last(times)
    grid = sort(unique(vcat(collect(p.dt:p.dt:T), times)))
    w, m, P = copy(prior.w), copy(prior.m), copy(prior.P)
    dev = zeros(length(w))
    out = Vector{GaussianMixture}(undef, length(times))
    ta = 0.0
    for tb in grid
        h = tb - ta
        if isfinite(p.τ)
            steps = [_entropy_step(model, m[j], P[j], h) for j in eachindex(w)]
            acc = [dev[j] + steps[j][1] for j in eachindex(w)]
            while length(w) + 2 <= p.max_components
                j = argmax(acc)
                acc[j] > p.τ || break
                gm = split_component(GaussianMixture(w, m, P), j, _whiten_direction(P[j], steps[j][2]);
                                     shrink = p.shrink)
                # children replace component j at positions j:j+2
                w, m, P = gm.w, gm.m, gm.P
                child = [_entropy_step(model, m[k], P[k], h) for k in j:j+2]
                steps = vcat(steps[1:j-1], child, steps[j+1:end])
                acc = vcat(acc[1:j-1], [c[1] for c in child], acc[j+1:end])
            end
            dev = acc
        end
        stepped = [_linear_step(model, m[j], P[j], h) for j in eachindex(w)]
        m, P = first.(stepped), last.(stepped)
        for i in findall(==(tb), times)
            out[i] = GaussianMixture(w, m, P)
        end
        ta = tb
    end
    return out
end

# P3/P4 criterion and split direction for one component over horizon h
function _horizon_criterion(p::HorizonSplit, model::DynamicsModel, m, P, h)
    sol = propagate_tensors(model, m, h)
    S = Matrix(cholesky(Symmetric(Matrix(P))).L)
    if p.measurement === nothing
        B = whitened_tensor(sol.Φ[end], sol.Ψ[end], S)
        return negentropy_leading(B), first(max_nonlinearity_direction(B))
    else
        A, Q, _ = measurement_tensors(p.measurement, sol.Φ[end], sol.Ψ[end], sol.x[end], S)
        return measurement_negentropy(A, Q), first(max_nonlinearity_direction(Q))
    end
end

# Split the worst component (by the criterion over horizon h) until every one is
# within τ or the cap is reached. Returns the mixture and each component's criterion.
function _decide_splits(p::HorizonSplit, model::DynamicsModel, gm::GaussianMixture, h)
    crit = [_horizon_criterion(p, model, gm.m[j], gm.P[j], h) for j in eachindex(gm.w)]
    while ncomponents(gm) + 2 <= p.max_components
        j = argmax(first.(crit))
        crit[j][1] > p.τ || break
        gm = split_component(gm, j, crit[j][2]; shrink = p.shrink)
        child = [_horizon_criterion(p, model, gm.m[k], gm.P[k], h) for k in j:j+2]
        crit = vcat(crit[1:j-1], child, crit[j+1:end])
    end
    return gm, first.(crit)
end

function _merge_redundant(gm::GaussianMixture, τ_merge)
    while ncomponents(gm) > 1
        n = ncomponents(gm)
        costs = [(merge_cost(gm, i, j), i, j) for i in 1:n for j in i+1:n]
        c, i, j = minimum(costs)
        c < τ_merge || break
        gm = merge_components(gm, i, j)
    end
    return gm
end

function propagate_policy(p::HorizonSplit, model::DynamicsModel, prior::GaussianMixture,
                          times::AbstractVector; epochs = Float64[])
    _check_times(times)
    T = last(times)
    D = vcat(0.0, sort(filter(e -> 0 < e < T, collect(float.(epochs)))))
    E = vcat(D[2:end], T)
    gm = prior
    out = Vector{GaussianMixture}(undef, length(times))
    for (d, e) in zip(D, E)
        p.τ_merge > 0 && (gm = _merge_redundant(gm, p.τ_merge))
        isfinite(p.τ) && (gm = first(_decide_splits(p, model, gm, e - d)))
        t = d
        for i in findall(tk -> d < tk <= e, times)
            gm = _linear_mixture(model, gm, times[i] - t)
            t = times[i]
            out[i] = gm
        end
        gm = _linear_mixture(model, gm, e - t)
    end
    return out
end
