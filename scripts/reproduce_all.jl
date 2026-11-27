# Regenerate every figure and results/summary.md from a clean checkout:
#
#   julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
#   julia --project=scripts -t auto scripts/reproduce_all.jl
#
# Each experiment script can also be run on its own.

const SCRIPTS = ["e1_reversibility.jl", "e2_ratchet.jl", "e3_predictor.jl", "e4_accuracy.jl"]

for s in SCRIPTS
    @info "running $s"
    t = @elapsed include(joinpath(@__DIR__, s))
    @info "finished $s in $(round(t; digits = 1)) s"
end
