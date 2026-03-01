# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    most_fractional_column(ctx, primal_values; tol=1e-6)

Find the column variable with the most fractional LP value (closest
to the midpoint between its floor and ceil). Returns `(var, sp_id)`
or `(nothing, nothing)` if all columns are integral.
"""
function most_fractional_column(
    ctx,
    primal_values::Dict{MOI.VariableIndex,Float64};
    tol::Float64 = 1e-6
)
    best_var = nothing
    best_sp_id = nothing
    best_frac = 1.0

    for (master_var, sp_id, _, _) in ColGen.columns(bp_pool(ctx))
        val = get(primal_values, master_var, 0.0)
        frac_part = val - floor(val)
        (frac_part < tol || frac_part > 1.0 - tol) && continue
        dist = abs(frac_part - 0.5)
        if dist < best_frac
            best_frac = dist
            best_var = master_var
            best_sp_id = sp_id
        end
    end
    return best_var, best_sp_id
end

"""
    create_branching_children(id_counter, parent, branch_var, branch_val,
                              sp_id, ctx, dual_bound)

Create left and right child nodes by branching on `branch_var`.
Left child: UB(branch_var) = floor(branch_val).
Right child: LB(branch_var) = ceil(branch_val).
Backward diffs restore original bounds (LB=0, UB=convexity_ub).
"""
function create_branching_children(
    id_counter, parent, branch_var, branch_val,
    sp_id, ctx, dual_bound
)
    floor_val = floor(branch_val)
    ceil_val = ceil(branch_val)
    _, conv_ub = ColGen.convexity_bounds(bp_decomp(ctx), sp_id)

    left_fwd = MathOptState.DomainChangeDiff(
        MathOptState.LowerBoundVarChange[],
        [MathOptState.UpperBoundVarChange(branch_var, floor_val)]
    )
    left_bwd = MathOptState.DomainChangeDiff(
        MathOptState.LowerBoundVarChange[],
        [MathOptState.UpperBoundVarChange(branch_var, conv_ub)]
    )

    right_fwd = MathOptState.DomainChangeDiff(
        [MathOptState.LowerBoundVarChange(branch_var, ceil_val)],
        MathOptState.UpperBoundVarChange[]
    )
    right_bwd = MathOptState.DomainChangeDiff(
        [MathOptState.LowerBoundVarChange(branch_var, 0.0)],
        MathOptState.UpperBoundVarChange[]
    )

    left = TreeSearch.child_node(
        id_counter, parent, left_fwd, left_bwd;
        dual_bound = dual_bound,
        user_data = BPNodeData()
    )
    right = TreeSearch.child_node(
        id_counter, parent, right_fwd, right_bwd;
        dual_bound = dual_bound,
        user_data = BPNodeData()
    )
    return [left, right]
end
