# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

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
    ws::ColGenWorkspace,
    phase::CGPhase,
    sps_db,
    mast_dual_sol::MasterDualSolution
)
    decomp = ws.decomp
    dual_values(cstr) = _dual_value(mast_dual_sol, cstr)

    # Verify dual solution consistency
    recomputed = _recompute_cost(mast_dual_sol.sol, master_model(decomp))
    stored = mast_dual_sol.sol.obj_value
    abs(recomputed - stored) < DUAL_CONSISTENCY_TOL || error(
        "Dual cost mismatch: recomputed=$recomputed stored=$stored"
    )

    # 1. Convexity contribution: Σₖ (ν⁺·W̄ₖ + ν⁻·Wₖ)
    #    A subproblem with a setup variable keeps its term: ν is in the
    #    reduced cost of the setup variable, hence in its SP_db.
    convexity_contrib = 0.0
    for sp_id in subproblem_ids(decomp)
        if !has_setup_variable(decomp, sp_id)
            conv_lb, conv_ub = convexity_bounds(decomp, sp_id)
            if has_convexity_ub(decomp, sp_id)
                ν_ub = dual_values(convexity_ub_ci(decomp, sp_id))
                convexity_contrib += ν_ub * conv_ub
            end
            if has_convexity_lb(decomp, sp_id)
                ν_lb = dual_values(convexity_lb_ci(decomp, sp_id))
                convexity_contrib += ν_lb * conv_lb
            end
        end
    end

    # 2. Subproblem contribution: Σₖ (SP_db + cost_phase(fₖ)) × multiplicity
    sense = is_minimization(decomp) ? 1 : -1
    sp_contrib = 0.0
    for (sp_id, sp_db) in sps_db
        fk = subproblem_fixed_cost(decomp, sp_id)
        sp_total = sp_db + _objective_cost(phase, fk)
        conv_lb, conv_ub = convexity_bounds(decomp, sp_id)
        multiplicity = (sense * sp_total < 0) ? conv_ub : conv_lb
        sp_contrib += sp_total * multiplicity
    end

    # 3. Subtract pure master variable bound duals
    #    (avoid double-counting with pure_master_contrib below).
    #    An infinite bound has no constraint, hence no dual: skipping it
    #    avoids 0 * Inf = NaN.
    pure_master_bound_contrib = 0.0
    for y_var in pure_master_variables(decomp)
        lb, ub = pure_master_bounds(decomp, y_var)
        if isfinite(lb)
            lb_ci = TaggedCI(y_var.id.value, VI_GEQ)
            lb_dual = get(
                mast_dual_sol.sol.constraint_duals, lb_ci, 0.0
            )
            pure_master_bound_contrib += lb_dual * lb
        end
        if isfinite(ub)
            ub_ci = TaggedCI(y_var.id.value, VI_LEQ)
            ub_dual = get(
                mast_dual_sol.sol.constraint_duals, ub_ci, 0.0
            )
            pure_master_bound_contrib += ub_dual * ub
        end
        # A fixed variable has a single constraint, y == lb == ub.
        if lb == ub
            eq_ci = TaggedCI(y_var.id.value, VI_EQ)
            eq_dual = get(
                mast_dual_sol.sol.constraint_duals, eq_ci, 0.0
            )
            pure_master_bound_contrib += eq_dual * lb
        end
    end

    # 4. Pure master variable contribution
    pure_master_contrib = compute_dual_bound_pure_master_contribution(
        decomp, dual_values
    )

    return mast_dual_sol.sol.obj_value - convexity_contrib -
        pure_master_bound_contrib + sp_contrib + pure_master_contrib
end
