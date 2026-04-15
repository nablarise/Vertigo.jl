# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    _objective_cost(::Phase0, col_cost) -> Float64
    _objective_cost(::Phase1, col_cost) -> Float64
    _objective_cost(::Phase2, col_cost) -> Float64

Objective coefficient for a new column variable in the master.
Phase 1 uses zero (only artificial variables carry cost).
"""
_objective_cost(::Union{Phase0,Phase2}, col_cost::Float64) = col_cost
_objective_cost(::Phase1, ::Float64) = 0.0

"""
    _insert_column!(ctx, phase, pricing_sol) → Union{MOI.VariableIndex, Nothing}

Insert a single column into the master. Returns the new column variable
index, or `nothing` if the column was already in the pool.
"""
function _insert_column!(
    ctx::ColGenContext, phase::CGPhase,
    pricing_sol::PricingPrimalSolution
)
    decomp = ctx.decomp
    model = master_model(decomp)

    sp_id = pricing_sol.sp_id
    sol = pricing_sol.solution

    has_column(ctx.pool, sp_id, sol) && return nothing

    col_cost = compute_column_original_cost(decomp, sp_id, sol)

    coupling_coeffs = compute_column_coupling_coefficients(
        decomp, sp_id, sol
    )
    # TODO: compute non-robust cut coefficients for new columns

    all_coeffs = Dict{TaggedCI,Float64}()
    for (tagged_ci, v) in coupling_coeffs
        all_coeffs[tagged_ci] = v
    end

    for bc in ctx.branching_constraints
        coeff = compute_branching_column_coefficient(
            decomp, bc.orig_var, sp_id, sol
        )
        if !iszero(coeff)
            all_coeffs[bc.constraint_index] = coeff
        end
    end

    for cut in ctx.robust_cuts
        coeff = 0.0
        for (sp_var_inner, val_inner) in nonzero_entries(sol)
            ov = mapped_original_var(decomp, sp_id, sp_var_inner)
            isnothing(ov) && continue
            c = get(cut.coefficients, ov, 0.0)
            if !iszero(c)
                coeff += c * val_inner
            end
        end
        if !iszero(coeff)
            all_coeffs[cut.constraint_index] = coeff
        end
    end

    if has_convexity_ub(decomp, sp_id)
        all_coeffs[convexity_ub_ci(decomp, sp_id)] = 1.0
    end
    if has_convexity_lb(decomp, sp_id)
        all_coeffs[convexity_lb_ci(decomp, sp_id)] = 1.0
    end

    obj_coeff = _objective_cost(phase, col_cost)
    col_var = add_variable!(model;
        lower_bound = 0.0,
        constraint_coeffs = all_coeffs,
        objective_coeff = obj_coeff
    )
    record_column!(ctx.pool, col_var, sp_id, sol, col_cost)

    return col_var
end

function insert_columns!(
    ctx::ColGenContext, phase::CGPhase,
    columns::GeneratedColumns
)
    cols_inserted = 0
    for pricing_sol in columns.collection
        col_var = _insert_column!(ctx, phase, pricing_sol)
        if !isnothing(col_var)
            cols_inserted += 1
        end
    end
    return cols_inserted
end
