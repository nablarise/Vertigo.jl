# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    BPEvaluator <: TreeSearch.AbstractNodeEvaluator

Node evaluator for branch-and-price. Runs column generation at each
node and returns CUTOFF, FEASIBLE, or BRANCH.
"""
struct BPEvaluator <: TreeSearch.AbstractNodeEvaluator end

function TreeSearch.evaluate!(
    ::BPEvaluator, space::BPSpace, node
)
    space.nodes_explored += 1
    cg_output = ColGen.run_column_generation(space.ctx)
    node.user_data = BPNodeData(cg_output)

    # Infeasible node
    if cg_output.status == ColGen.master_infeasible ||
       cg_output.status == ColGen.subproblem_infeasible
        return TreeSearch.CUTOFF
    end

    # Update node dual bound
    db = cg_output.incumbent_dual_bound
    if !isnothing(db)
        node.dual_bound = db
        if ColGen.is_minimization(space.ctx)
            space.best_dual_bound = max(space.best_dual_bound, db)
        else
            space.best_dual_bound = min(space.best_dual_bound, db)
        end
    end

    # Prune by bound
    if !isnothing(space.incumbent) && !isnothing(db)
        if ColGen.is_minimization(space.ctx)
            db >= space.incumbent.obj_value - space.tol &&
                return TreeSearch.CUTOFF
        else
            db <= space.incumbent.obj_value + space.tol &&
                return TreeSearch.CUTOFF
        end
    end

    # Detect new IP-feasible solution from CG
    ip_sol = bp_ip_incumbent(space.ctx)
    if !isnothing(ip_sol) && ip_sol !== space.last_ip_incumbent
        space.last_ip_incumbent = ip_sol
        if isnothing(space.incumbent) ||
           (ColGen.is_minimization(space.ctx) ?
                ip_sol.obj_value < space.incumbent.obj_value - space.tol :
                ip_sol.obj_value > space.incumbent.obj_value + space.tol)
            space.incumbent = ip_sol
            return TreeSearch.FEASIBLE
        end
    end

    return TreeSearch.BRANCH
end
