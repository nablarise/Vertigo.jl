# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# Pricing oracle for a user-overridden GAP subproblem: the single-machine
# knapsack, solved exactly by dynamic programming.
#
# Vertigo builds no model for such a subproblem. It only writes the current
# reduced cost of every task into the pricer (`MOI.modify`) and then asks it
# to price (`CG.optimize_pricing_problem!`).

using Vertigo
import MathOptInterface as MOI

const CG = Vertigo.ColGen

mutable struct KnapsackPricer
    weight::Vector{Int}                 # task weights on this machine
    capacity::Int
    sp_vars::Vector{MOI.VariableIndex}  # sp_vars[t]: Vertigo's variable for task t
    red_cost::Vector{Float64}           # current reduced cost of every task
end

function KnapsackPricer(weight, capacity, sp_vars)
    all(isinteger, weight) && isinteger(capacity) ||
        error("the knapsack DP needs integer weights and capacity")
    return KnapsackPricer(Int.(weight), Int(capacity), sp_vars, zeros(length(sp_vars)))
end

# Vertigo calls this whenever a task's reduced cost changes.
function MOI.modify(
    p::KnapsackPricer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    change::MOI.ScalarCoefficientChange{Float64}
)
    p.red_cost[findfirst(==(change.variable), p.sp_vars)] = change.new_coefficient
    return nothing
end

"""
    solve_knapsack(p) -> (value, tasks)

Minimize the total reduced cost of a set of tasks whose weights fit in the capacity
(0/1 knapsack). `best[c + 1]` is the lowest cost reachable with a load of at most `c`
using the tasks seen so far; going through the tasks one by one, a task is added to a
load `c` when that beats what `best` already holds. `take[t, c + 1]` remembers that, so the
chosen tasks can be read back from the last task to the first. Only tasks with a negative
reduced cost can lower the total, so the others are skipped.
"""
function solve_knapsack(p::KnapsackPricer)
    n = length(p.weight)
    best = zeros(p.capacity + 1)
    take = falses(n, p.capacity + 1)
    for t in 1:n
        p.red_cost[t] < 0 || continue
        for c in p.capacity:-1:p.weight[t]
            with_t = best[c - p.weight[t] + 1] + p.red_cost[t]
            if with_t < best[c + 1]
                best[c + 1] = with_t
                take[t, c + 1] = true
            end
        end
    end

    tasks, c = Int[], p.capacity
    for t in n:-1:1
        if take[t, c + 1]
            push!(tasks, t)
            c -= p.weight[t]
        end
    end
    return best[p.capacity + 1], sort(tasks)
end

function CG.get_pricing_subprob_optimizer(::CG.ExactStage, sp::CG.PricingSubproblem{<:KnapsackPricer})
    return CG.moi_pricing_sp(sp)
end

function CG.optimize_pricing_problem!(
    ws::CG.ColGenWorkspace, sp_id, ::CG.PricingSubproblem, p::KnapsackPricer,
    mast_dual_sol, _stab_changes_mast_dual_sol
)
    decomp = ws.decomp
    ν = CG._get_convexity_dual(mast_dual_sol, convexity_lb_pairs(decomp), sp_id) +
        CG._get_convexity_dual(mast_dual_sol, convexity_ub_pairs(decomp), sp_id)
    fk = subproblem_fixed_cost(decomp, sp_id)

    sp_obj, tasks = solve_knapsack(p)

    # The column's reduced cost in the master: subproblem cost + fixed cost - convexity dual.
    rc = sp_obj + fk - ν
    sol = Vertigo.Reformulation._SpSolution(sp_id, rc, [(p.sp_vars[t], 1.0) for t in tasks])
    column = CG.PricingPrimalSolution(sp_id, sol, rc < -CG.RC_IMPROVING_TOL)

    # The DP is exact: its value is both the primal and the dual bound of the subproblem.
    return CG.PricingSolution(false, false, rc, sp_obj, [column])
end
