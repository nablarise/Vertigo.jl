# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    BPEvaluator <: TreeSearch.AbstractNodeEvaluator

Node evaluator for branch-and-price. Runs column generation at each
node and returns CUTOFF, FEASIBLE, or BRANCH.
"""
struct BPEvaluator <: TreeSearch.AbstractNodeEvaluator end

function _rebuild_branching_constraints!(space::BranchCutPriceWorkspace)
    bcs = bp_branching_constraints(space.ws)
    empty!(bcs)
    for (cut_id, ci) in space.cut_helper.active_cuts
        ov = space.branching_cut_info[cut_id]
        push!(bcs, ColGen.ActiveBranchingConstraint(ci, ov))
    end
    _update_column_branching_coefficients!(space)
    return
end

# A branching constraint is re-created, when its node is entered, with the
# columns that existed when it was created. The columns generated since, in
# other nodes, get their coefficient here: the pricing assumes it, and without
# it a column that breaks the constraint is allowed, so the pricing finds it
# again and it is refused as a duplicate.
function _update_column_branching_coefficients!(space::BranchCutPriceWorkspace)
    ws = space.ws
    decomp = bp_decomp(ws)
    model = bp_master_model(ws)
    for bc in bp_branching_constraints(ws)
        for (col_var, rec) in columns(bp_pool(ws))
            coeff = compute_branching_column_coefficient(
                decomp, bc.orig_var, column_sp_id(rec), rec.solution
            )
            if !iszero(coeff)
                with_typed_ci(bc.constraint_index) do ci
                    MOI.modify(
                        model, ci, MOI.ScalarCoefficientChange(col_var, coeff)
                    )
                end
            end
        end
    end
    return
end

function _should_prune(space::BranchCutPriceWorkspace, db)::Bool
    isnothing(space.incumbent) && return false
    isnothing(db) && return false
    if is_minimization(space.ws)
        return db >= space.incumbent.obj_value - space.tol
    end
    return db <= space.incumbent.obj_value + space.tol
end

function _is_improving_incumbent(
    space::BranchCutPriceWorkspace, sol::ColGen.MasterIpPrimalSol
)::Bool
    isnothing(space.incumbent) && return true
    if is_minimization(space.ws)
        return sol.obj_value < space.incumbent.obj_value - space.tol
    end
    return sol.obj_value > space.incumbent.obj_value + space.tol
end

# An unbranchable node is never popped again, so the new incumbent is the
# only thing that can prune it.
function _set_improving_incumbent!(space::BranchCutPriceWorkspace, sol)
    space.incumbent = sol
    filter!(
        node_bound -> !_should_prune(space, node_bound.second),
        space.unbranchable_node_bounds
    )
    return
end

# Record a new integer solution found by column generation as the incumbent,
# and return whether it closes the node. Column generation can meet it before
# converging, so the node is closed only if its bound reaches the incumbent.
function _close_on_ip_incumbent!(space::BranchCutPriceWorkspace, node, ip_sol)
    closed = false
    if !isnothing(ip_sol) && ip_sol !== space.last_ip_incumbent
        space.last_ip_incumbent = ip_sol
        if _is_improving_incumbent(space, ip_sol)
            _set_improving_incumbent!(space, ip_sol)
            if _should_prune(space, node.dual_bound)
                delete!(space.open_node_bounds, node.id)
                _recompute_global_dual_bound!(space)
                closed = true
            end
        end
    end
    return closed
end

# The bound a node inherits from its parent stays valid: a column generation
# stopped early (time limit) can end with a weaker one.
function _tighter_bound(space::BranchCutPriceWorkspace, inherited, found)
    if !isfinite(inherited)
        return found
    end
    return is_minimization(space.ws) ? max(inherited, found) : min(inherited, found)
end

function _set_incumbent_bound!(space::BranchCutPriceWorkspace)
    bp_set_ip_primal_bound!(
        space.ws,
        isnothing(space.incumbent) ?
            nothing : space.incumbent.obj_value
    )
    return
end

function TreeSearch.evaluate!(
    ::BPEvaluator, space::BranchCutPriceWorkspace, node
)
    space.nodes_explored += 1

    _set_incumbent_bound!(space)
    _rebuild_branching_constraints!(space)

    # First CG + cut separation (unconditional)
    cg_output = ColGen.run_column_generation(space.ws)
    nb_cuts = _separate_and_add_cuts!(space, cg_output)
    lp = _master_lp_obj(cg_output)
    round = 0
    prev_lp = Inf

    # Cut-and-column-generation loop
    while !stop_cutcolgen(
        space.cutcolgen_ctx, round, nb_cuts,
        cg_output.status, prev_lp, lp
    )
        prev_lp = lp
        round += 1
        _set_incumbent_bound!(space)
        cg_output = ColGen.run_column_generation(space.ws)
        nb_cuts = _separate_and_add_cuts!(space, cg_output)
        lp = _master_lp_obj(cg_output)
    end

    # Root LP value after valid inequality separation
    # (`nothing` if the root LP was infeasible or never solved).
    if space.nodes_explored == 1
        space.root_lp_value = cg_output.master_lp_obj
    end

    if isnothing(node.user_data)
        node.user_data = BPNodeData()
    end
    node.user_data.cg_output = cg_output
    on_node_evaluated(
        space.branching_strategy, space, node, cg_output
    )

    # Infeasible node
    if cg_output.status == ColGen.master_infeasible ||
       cg_output.status == ColGen.subproblem_infeasible ||
       cg_output.status == ColGen.ip_pruned
        delete!(space.open_node_bounds, node.id)
        _recompute_global_dual_bound!(space)
        return TreeSearch.CUTOFF
    end

    # Update node dual bound
    db = cg_output.incumbent_dual_bound
    if !isnothing(db)
        node.dual_bound = _tighter_bound(space, node.dual_bound, db)
    end

    # Prune by bound
    if _should_prune(space, node.dual_bound)
        delete!(space.open_node_bounds, node.id)
        _recompute_global_dual_bound!(space)
        return TreeSearch.CUTOFF
    end

    # The time ran out inside column generation: the master LP does not
    # hold the last columns, so the node can be neither branched nor read.
    # Its bound stays among the open ones and the search stops.
    if cg_output.status == ColGen.time_limit
        space.open_node_bounds[node.id] = node.dual_bound
        _recompute_global_dual_bound!(space)
        return TreeSearch.CUTOFF
    end

    if _close_on_ip_incumbent!(space, node, cg_output.ip_incumbent)
        return TreeSearch.FEASIBLE
    end

    # The node stays open until it is branched, so the global dual bound
    # (read by the logger right after this call) accounts for it.
    space.open_node_bounds[node.id] = node.dual_bound
    _recompute_global_dual_bound!(space)

    # Restricted master IP heuristic
    if space.rmp_heuristic && !_time_is_up(space)
        rmp_sol = solve_restricted_master_ip!(space, cg_output)
        if !isnothing(rmp_sol) &&
                _is_improving_incumbent(space, rmp_sol)
            _set_improving_incumbent!(space, rmp_sol)
            _recompute_global_dual_bound!(space)
        end
    end

    return TreeSearch.BRANCH
end
