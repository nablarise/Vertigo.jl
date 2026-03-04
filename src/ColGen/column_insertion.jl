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

        all_coeffs = Dict{TaggedCI,Float64}()
        for (tagged_ci, v) in coupling_coeffs
            all_coeffs[tagged_ci] = v
        end
        for (k, v) in cut_coeffs
            all_coeffs[TaggedCI(k)] = v
        end

        for bc in ctx.branching_constraints
            coeff = compute_branching_column_coefficient(
                decomp, bc.orig_var, sp_id, sol
            )
            if !iszero(coeff)
                all_coeffs[TaggedCI(bc.constraint_index)] = coeff
            end
        end

        if haskey(ctx.convexity_ub, sp_id)
            all_coeffs[TaggedCI(ctx.convexity_ub[sp_id])] = 1.0
        end
        if haskey(ctx.convexity_lb, sp_id)
            all_coeffs[TaggedCI(ctx.convexity_lb[sp_id])] = 1.0
        end

        obj_coeff = _objective_cost(phase, col_cost)
        col_var = add_variable!(model;
            lower_bound = 0.0,
            constraint_coeffs = all_coeffs,
            objective_coeff = obj_coeff
        )
        record_column!(ctx.pool, col_var, sp_id, sol, col_cost)

        cols_inserted += 1
    end

    return cols_inserted
end
