# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function compute_dual_bound(
    ctx::ColGenContext,
    ::Union{Phase0,Phase1,Phase2},
    sps_db,
    mast_dual_sol::MasterDualSolution
)
    decomp = ctx.decomp
    dual_values(cstr) = _dual_value(mast_dual_sol, cstr)

    # Verify dual solution consistency
    recomputed = recompute_cost(mast_dual_sol.sol, ctx.master_model)
    stored = mast_dual_sol.sol.obj_value
    @assert abs(recomputed - stored) < 1e-4 "Dual cost mismatch: recomputed=$recomputed stored=$stored"

    # 1. Convexity contribution: Σₖ (ν⁺·W̄ₖ + ν⁻·Wₖ)
    convexity_contrib = 0.0
    for sp_id in subproblem_ids(decomp)
        conv_lb, conv_ub = convexity_bounds(decomp, sp_id)
        if haskey(ctx.convexity_ub, sp_id)
            ν_ub = dual_values(ctx.convexity_ub[sp_id])
            convexity_contrib += ν_ub * conv_ub
        end
        if haskey(ctx.convexity_lb, sp_id)
            ν_lb = dual_values(ctx.convexity_lb[sp_id])
            convexity_contrib += ν_lb * conv_lb
        end
    end

    # 2. Subproblem contribution: Σₖ SP_db × multiplicity
    sense = is_minimization(decomp) ? 1 : -1
    sp_contrib = 0.0
    for (sp_id, sp_db) in sps_db
        conv_lb, conv_ub = convexity_bounds(decomp, sp_id)
        multiplicity = (sense * sp_db < 0) ? conv_ub : conv_lb
        sp_contrib += sp_db * multiplicity
    end

    # 3. Pure master variable contribution (0 for GAP)
    y_contrib = compute_dual_bound_y_contribution(decomp, dual_values)

    return mast_dual_sol.sol.obj_value - convexity_contrib + sp_contrib + y_contrib
end
