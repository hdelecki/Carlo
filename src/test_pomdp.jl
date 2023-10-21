using POMDPs
using POMDPTools
using POMDPModels
using PointBasedValueIteration
using POMDPTools: SparseCat, Deterministic
using Plots
using BSON: @save, @load



include("carlopomdp.jl")
include("likelihood_learning.jl")

function POMDPs.initialstate(pomdp::CarloPOMDP)
    ego = [4, 8, 6, 5, 2]
    rival = collect(1:8)
    rival_vel = [:within_limit, :above_limit]
    rival_itn = [:left, :right, :straight]
    results = Iterators.product([ego, rival, rival_vel, rival_itn]...) |> collect |> vec
    s0 = [CarloDiscreteState(item...) for item in results]
    return SparseCat(s0, ones(length(s0))/length(s0))
end

pomdp = CarloPOMDP(discount=0.9)
tab_pomdp = tabulate(pomdp; dir="../sw-pomdp/")
@save "sw_pomdp.bson" tab_pomdp

# @load "./src/tab_pomdp.bson" tab_pomdp
# Onew = deepcopy(tab_pomdp.O)
# Onew[isnan.(Onew)] .= 0.0
# Tnew = deepcopy(tab_pomdp.T)
# Rnew = deepcopy(tab_pomdp.R)

# tpomdp = TabularPOMDP(Tnew, Rnew, Onew, 0.9)

# heatmap(isnan.(tab_pomdp.O[:, 3, :]))
# tab_pomdp.O[isnan.(tab_pomdp)] .= 0.0


solver = PBVISolver(max_iterations=10, ϵ=5e-1, verbose=true)
# sparse_pomdp = SparseTabularPOMDP(pomdp)
policy = solve(solver, tab_pomdp)
# @save "sw_policy.bson" policy tpomdp


rsim = RolloutSimulator(max_steps=1000)
r = simulate(rsim, pomdp, policy)

mean([simulate(rsim, pomdp, policy) for _=1:100])