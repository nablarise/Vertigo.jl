# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    most_fractional_original_variable(ctx, primal_values; tol=1e-6)

Project the master LP solution to original-variable space and find the
most fractional original variable (closest to 0.5). Returns
`(orig_var, x_val)` or `(nothing, nothing)` if all are integral.
"""
function most_fractional_original_variable(
    ctx,
    primal_values::Dict{MOI.VariableIndex,Float64};
    tol::Float64 = 1e-6
)
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)
    x_values = project_to_original(
        decomp, pool, v -> get(primal_values, v, 0.0)
    )

    best_var = nothing
    best_val = 0.0
    best_dist = 1.0

    for (orig_var, x_val) in x_values
        frac_part = x_val - floor(x_val)
        (frac_part < tol || frac_part > 1.0 - tol) && continue
        dist = abs(frac_part - 0.5)
        if dist < best_dist
            best_dist = dist
            best_var = orig_var
            best_val = x_val
        end
    end
    return best_var, best_val
end

"""
    create_branching_children(id_counter, parent, orig_var, x_val,
                              ctx, dual_bound, cut_tracker)

Create left and right child nodes by branching on original variable
`orig_var`. Each child adds a robust branching constraint as a
`LocalCut` via tuple diffs `(DomainChangeDiff, LocalCutChangeDiff)`.

Left child:  Σ_j λ_j · a_{i,j} ≤ ⌊x_val⌋
Right child: Σ_j λ_j · a_{i,j} ≥ ⌈x_val⌉
"""
function create_branching_children(
    id_counter, parent, orig_var, x_val,
    ctx, dual_bound, cut_tracker
)
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)
    floor_val = floor(x_val)
    ceil_val = ceil(x_val)

    # Build constraint terms for all existing columns
    terms = MOI.ScalarAffineTerm{Float64}[]
    for (col_var, rec) in columns(pool)
        coeff = compute_branching_column_coefficient(
            decomp, orig_var, column_sp_id(rec), rec.solution
        )
        if !iszero(coeff)
            push!(terms, MOI.ScalarAffineTerm(coeff, col_var))
        end
    end

    # Left child: ≤ floor(x_val)
    id_left = MathOptState.next_id!(cut_tracker)
    left_cut = MathOptState.LocalCut(
        id_left, terms, MOI.LessThan(floor_val)
    )
    left_fwd = (
        MathOptState.DomainChangeDiff(),
        MathOptState.LocalCutChangeDiff(
            [MathOptState.AddLocalCutChange(left_cut)],
            MathOptState.RemoveLocalCutChange[]
        )
    )
    left_bwd = (
        MathOptState.DomainChangeDiff(),
        MathOptState.LocalCutChangeDiff(
            MathOptState.AddLocalCutChange[],
            [MathOptState.RemoveLocalCutChange(left_cut)]
        )
    )

    # Right child: ≥ ceil(x_val)
    id_right = MathOptState.next_id!(cut_tracker)
    right_cut = MathOptState.LocalCut(
        id_right, terms, MOI.GreaterThan(ceil_val)
    )
    right_fwd = (
        MathOptState.DomainChangeDiff(),
        MathOptState.LocalCutChangeDiff(
            [MathOptState.AddLocalCutChange(right_cut)],
            MathOptState.RemoveLocalCutChange[]
        )
    )
    right_bwd = (
        MathOptState.DomainChangeDiff(),
        MathOptState.LocalCutChangeDiff(
            MathOptState.AddLocalCutChange[],
            [MathOptState.RemoveLocalCutChange(right_cut)]
        )
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
    return [left, right], [(id_left, orig_var), (id_right, orig_var)]
end
