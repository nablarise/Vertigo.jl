# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    _add_column_variable!(model, all_coeffs, objective_coeff) -> MOI.VariableIndex

Add a new non-negative column variable to the master MOI model.
Avoids the type constraint on add_variable! by using MOI calls directly.
"""
function _add_column_variable!(model, all_coeffs, objective_coeff::Float64)
    var = MOI.add_variable(model)
    MOI.add_constraint(model, var, MOI.GreaterThan(0.0))

    if !iszero(objective_coeff)
        MOI.modify(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarCoefficientChange(var, objective_coeff)
        )
    end

    for (cstr_ref, coeff) in all_coeffs
        if !iszero(coeff)
            MOI.modify(model, cstr_ref, MOI.ScalarCoefficientChange(var, coeff))
        end
    end

    return var
end

function insert_columns!(
    ctx::ColGenContext, ::Union{Phase0,Phase2}, columns::GeneratedColumns
)
    model = ctx.master_model
    decomp = ctx.decomp
    cols_inserted = 0

    for pricing_sol in columns.collection
        sp_id = pricing_sol.sp_id
        sol = pricing_sol.solution

        # Skip duplicate columns
        has_column(ctx.pool, sp_id, sol) && continue

        # 1. Original column cost via Decomposition
        col_cost = compute_column_original_cost(decomp, sp_id, sol)

        # 2. Coupling constraint coefficients via Decomposition
        coupling_coeffs = compute_column_coupling_coefficients(decomp, sp_id, sol)

        # 3. Non-robust cut coefficients (empty unless cuts active)
        cut_coeffs = compute_column_cut_coefficients(ctx.cuts, sol)

        # 4. Merge all constraint coefficients
        all_coeffs = Dict{Any,Float64}()
        for (k, v) in coupling_coeffs; all_coeffs[k] = v; end
        for (k, v) in cut_coeffs; all_coeffs[k] = v; end

        # 5. Convexity constraint membership (coefficient = 1.0)
        if haskey(ctx.convexity_ub, sp_id)
            all_coeffs[ctx.convexity_ub[sp_id]] = 1.0
        end
        if haskey(ctx.convexity_lb, sp_id)
            all_coeffs[ctx.convexity_lb[sp_id]] = 1.0
        end

        # 6. Add column variable to master
        col_var = _add_column_variable!(model, all_coeffs, col_cost)

        # 7. Record in pool
        record_column!(ctx.pool, col_var, sp_id, sol, col_cost)

        cols_inserted += 1
    end

    return cols_inserted
end

function insert_columns!(
    ctx::ColGenContext, ::Phase1, columns::GeneratedColumns
)
    model = ctx.master_model
    decomp = ctx.decomp
    cols_inserted = 0

    for pricing_sol in columns.collection
        sp_id = pricing_sol.sp_id
        sol = pricing_sol.solution

        # Skip duplicate columns
        has_column(ctx.pool, sp_id, sol) && continue

        # 1. Original column cost for pool recording
        col_cost = compute_column_original_cost(decomp, sp_id, sol)

        # 2. Coupling constraint coefficients via Decomposition
        coupling_coeffs = compute_column_coupling_coefficients(decomp, sp_id, sol)

        # 3. Non-robust cut coefficients (empty unless cuts active)
        cut_coeffs = compute_column_cut_coefficients(ctx.cuts, sol)

        # 4. Merge all constraint coefficients
        all_coeffs = Dict{Any,Float64}()
        for (k, v) in coupling_coeffs; all_coeffs[k] = v; end
        for (k, v) in cut_coeffs; all_coeffs[k] = v; end

        # 5. Convexity constraint membership (coefficient = 1.0)
        if haskey(ctx.convexity_ub, sp_id)
            all_coeffs[ctx.convexity_ub[sp_id]] = 1.0
        end
        if haskey(ctx.convexity_lb, sp_id)
            all_coeffs[ctx.convexity_lb[sp_id]] = 1.0
        end

        # 6. Add column with zero objective cost (Phase1 minimises art vars only)
        col_var = _add_column_variable!(model, all_coeffs, 0.0)

        # 7. Record original cost in pool for use in Phase2
        record_column!(ctx.pool, col_var, sp_id, sol, col_cost)

        cols_inserted += 1
    end

    return cols_inserted
end
