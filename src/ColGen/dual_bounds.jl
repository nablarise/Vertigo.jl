# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

_rhs(s::MOI.LessThan) = s.upper
_rhs(s::MOI.GreaterThan) = s.lower
_rhs(s::MOI.EqualTo) = s.value
_rhs(s) = error("unsupported constraint set type: $(typeof(s))")

function _recompute_cost(dual_sol::DualMoiSolution, model)::Float64
    total_cost = 0.0
    for (tagged, dual_value) in dual_sol.constraint_duals
        with_typed_ci(tagged) do ci
            cset = MOI.get(model, MOI.ConstraintSet(), ci)
            total_cost += dual_value * _rhs(cset)
        end
    end
    obj_fn = MOI.get(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
    )
    total_cost += obj_fn.constant
    return total_cost
end

function compute_dual_bound(
    ctx::ColGenContext,
    ::Union{Phase0,Phase1,Phase2},
    sps_db,
    mast_dual_sol::MasterDualSolution
)
    decomp = ctx.decomp
    dual_values(cstr) = _dual_value(mast_dual_sol, cstr)

    # Verify dual solution consistency
    recomputed = _recompute_cost(mast_dual_sol.sol, ctx.master_model)
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

    # 2. Subproblem contribution: Σₖ (SP_db + fₖ) × multiplicity
    sense = is_minimization(decomp) ? 1 : -1
    sp_contrib = 0.0
    for (sp_id, sp_db) in sps_db
        fk = subproblem_fixed_cost(decomp, sp_id)
        sp_total = sp_db + fk
        conv_lb, conv_ub = convexity_bounds(decomp, sp_id)
        multiplicity = (sense * sp_total < 0) ? conv_ub : conv_lb
        sp_contrib += sp_total * multiplicity
    end

    # 3. Subtract pure master variable bound duals
    #    (avoid double-counting with pure_master_contrib below)
    pure_master_bound_contrib = 0.0
    for y_var in pure_master_variables(decomp)
        lb, ub = pure_master_bounds(decomp, y_var)
        lb_ci = TaggedCI(y_var.id.value, VI_GEQ)
        ub_ci = TaggedCI(y_var.id.value, VI_LEQ)
        lb_dual = get(
            mast_dual_sol.sol.constraint_duals, lb_ci, 0.0
        )
        ub_dual = get(
            mast_dual_sol.sol.constraint_duals, ub_ci, 0.0
        )
        pure_master_bound_contrib += lb_dual * lb + ub_dual * ub
    end

    # 4. Pure master variable contribution
    pure_master_contrib = compute_dual_bound_pure_master_contribution(
        decomp, dual_values
    )

    return mast_dual_sol.sol.obj_value - convexity_contrib -
        pure_master_bound_contrib + sp_contrib + pure_master_contrib
end
