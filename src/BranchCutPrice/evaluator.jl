# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    BPEvaluator <: TreeSearch.AbstractNodeEvaluator

Node evaluator for branch-and-price. Runs column generation at each
node and returns CUTOFF, FEASIBLE, or BRANCH.
"""
struct BPEvaluator <: TreeSearch.AbstractNodeEvaluator end

function _rebuild_branching_constraints!(space::BPSpace)
    bcs = bp_branching_constraints(space.ctx)
    empty!(bcs)
    for (cut_id, ci) in space.cut_helper.active_cuts
        ov = space.branching_cut_info[cut_id]
        push!(bcs, ColGen.ActiveBranchingConstraint(ci, ov))
    end
    return
end

function _should_prune(space::BPSpace, db)::Bool
    isnothing(space.incumbent) && return false
    isnothing(db) && return false
    if is_minimization(space.ctx)
        return db >= space.incumbent.obj_value - space.tol
    end
    return db <= space.incumbent.obj_value + space.tol
end

function _is_improving_incumbent(
    space::BPSpace, sol::ColGen.MasterIpPrimalSol
)::Bool
    isnothing(space.incumbent) && return true
    if is_minimization(space.ctx)
        return sol.obj_value < space.incumbent.obj_value - space.tol
    end
    return sol.obj_value > space.incumbent.obj_value + space.tol
end

function TreeSearch.evaluate!(
    ::BPEvaluator, space::BPSpace, node
)
    space.nodes_explored += 1
    delete!(space.open_node_bounds, node.id)

    # Pass incumbent bound to CG for early pruning
    raw_ctx = space.ctx isa ColGen.ColGenLoggerContext ?
        space.ctx.inner : space.ctx
    raw_ctx.ip_primal_bound = isnothing(space.incumbent) ?
        nothing : space.incumbent.obj_value

    _rebuild_branching_constraints!(space)
    cg_output = ColGen.run_column_generation(space.ctx)
    node.user_data = BPNodeData(cg_output)

    # Infeasible node
    if cg_output.status == ColGen.master_infeasible ||
       cg_output.status == ColGen.subproblem_infeasible
        _recompute_global_dual_bound!(space)
        return TreeSearch.CUTOFF
    end

    # Update node dual bound
    db = cg_output.incumbent_dual_bound
    if !isnothing(db)
        node.dual_bound = db
    end

    # Prune by bound
    if _should_prune(space, db)
        _recompute_global_dual_bound!(space)
        return TreeSearch.CUTOFF
    end

    # Detect new IP-feasible solution from CG
    ip_sol = cg_output.ip_incumbent
    if !isnothing(ip_sol) && ip_sol !== space.last_ip_incumbent
        space.last_ip_incumbent = ip_sol
        if _is_improving_incumbent(space, ip_sol)
            space.incumbent = ip_sol
            _recompute_global_dual_bound!(space)
            return TreeSearch.FEASIBLE
        end
    end

    # Restricted master IP heuristic
    if space.rmp_heuristic
        rmp_sol = solve_restricted_master_ip!(space, cg_output)
        if !isnothing(rmp_sol) &&
                _is_improving_incumbent(space, rmp_sol)
            space.incumbent = rmp_sol
            _recompute_global_dual_bound!(space)
        end
    end

    return TreeSearch.BRANCH
end
