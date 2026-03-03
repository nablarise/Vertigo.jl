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

"""
    _objective_cost(::Phase0, col_cost) -> Float64
    _objective_cost(::Phase1, col_cost) -> Float64
    _objective_cost(::Phase2, col_cost) -> Float64

Objective coefficient for a new column variable in the master.
Phase 1 uses zero (only artificial variables carry cost).
"""
_objective_cost(::Union{Phase0,Phase2}, col_cost::Float64) = col_cost
_objective_cost(::Phase1, ::Float64) = 0.0

function insert_columns!(
    ctx::ColGenContext, phase::Union{Phase0,Phase1,Phase2},
    columns::GeneratedColumns
)
    model = ctx.master_model
    decomp = ctx.decomp
    cols_inserted = 0

    for pricing_sol in columns.collection
        sp_id = pricing_sol.sp_id
        sol = pricing_sol.solution

        has_column(ctx.pool, sp_id, sol) && continue

        col_cost = compute_column_original_cost(decomp, sp_id, sol)

        coupling_coeffs = compute_column_coupling_coefficients(
            decomp, sp_id, sol
        )
        cut_coeffs = compute_column_cut_coefficients(ctx.cuts, sol)

        all_coeffs = Dict{Any,Float64}()
        for (k, v) in coupling_coeffs; all_coeffs[k] = v; end
        for (k, v) in cut_coeffs; all_coeffs[k] = v; end

        for bc in ctx.branching_constraints
            coeff = compute_branching_column_coefficient(
                decomp, bc.orig_var, sp_id, sol
            )
            if !iszero(coeff)
                all_coeffs[bc.constraint_index] = coeff
            end
        end

        if haskey(ctx.convexity_ub, sp_id)
            all_coeffs[ctx.convexity_ub[sp_id]] = 1.0
        end
        if haskey(ctx.convexity_lb, sp_id)
            all_coeffs[ctx.convexity_lb[sp_id]] = 1.0
        end

        obj_coeff = _objective_cost(phase, col_cost)
        col_var = _add_column_variable!(model, all_coeffs, obj_coeff)
        record_column!(ctx.pool, col_var, sp_id, sol, col_cost)

        cols_inserted += 1
    end

    return cols_inserted
end
