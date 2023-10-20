using POMDPs
using POMDPTools
using POMDPModels
using PointBasedValueIteration
using POMDPTools: SparseCat, Deterministic
using Plots
using BSON: @save, @load



include("carlopomdp.jl")
include("likelihood_learning.jl")

pomdp = CarloPOMDP()
tab_pomdp = tabulate(pomdp; dir="../s-w-pomdp/")
# @save "tab_pomdp.bson" tab_pomdp

# @load "./src/tab_pomdp.bson" tab_pomdp
Onew = deepcopy(tab_pomdp.O)
Onew[isnan.(Onew)] .= 0.0
Tnew = deepcopy(tab_pomdp.T)
Rnew = deepcopy(tab_pomdp.R)

tpomdp = TabularPOMDP(Tnew, Rnew, Onew, 0.9)

# heatmap(isnan.(tab_pomdp.O[:, 3, :]))
# tab_pomdp.O[isnan.(tab_pomdp)] .= 0.0


solver = PBVISolver(max_iterations=15, ϵ=1e-1, verbose=true)
policy = solve(solver, tpomdp)
@save "sw_policy.bson" policy tpomdp