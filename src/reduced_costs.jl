# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct ReducedCosts
    values::Dict{Any,Dict{Any,Float64}}
end

"""
    _dual_value(dual_sol, cstr_idx) -> Float64

Extract dual value for a specific constraint index from a MasterDualSolution.
Returns 0.0 if the constraint type or index is not present.
"""
function _dual_value(dual_sol::MasterDualSolution, cstr_idx)
    d = get(dual_sol.sol.constraint_duals, typeof(cstr_idx), nothing)
    isnothing(d) && return 0.0
    return get(d, cstr_idx.value, 0.0)
end

function compute_reduced_costs!(
    ctx::ColGenContext, ::MixedPhase1and2, mast_dual_sol::MasterDualSolution
)
    decomp = ctx.decomp
    dual_values(cstr) = _dual_value(mast_dual_sol, cstr)

    reduced_costs_dict = Dict{Any,Dict{Any,Float64}}()
    for sp_id in subproblem_ids(decomp)
        sp_rc = Dict{Any,Float64}()
        for sp_var in subproblem_variables(decomp, sp_id)
            rc = original_cost(decomp, sp_id, sp_var)
            for entry in coupling_coefficients(decomp, sp_id, sp_var)
                rc -= entry.coefficient * dual_values(entry.constraint_id)
            end
            rc -= total_cut_dual_contribution(ctx.cuts, sp_id, sp_var)
            sp_rc[sp_var] = rc
        end
        reduced_costs_dict[sp_id] = sp_rc
    end

    return ReducedCosts(reduced_costs_dict)
end

function update_reduced_costs!(ctx::ColGenContext, ::MixedPhase1and2, red_costs::ReducedCosts)
    for (sp_id, sp_rc) in red_costs.values
        sp_model = ctx.sp_models[sp_id]
        for (var_index, rc_value) in sp_rc
            MOI.modify(
                sp_model,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarCoefficientChange(var_index, rc_value)
            )
        end
    end
    return nothing
end

compute_sp_init_db(ctx::ColGenContext, _) = is_minimization(ctx) ? -Inf : Inf
compute_sp_init_pb(ctx::ColGenContext, _) = is_minimization(ctx) ? Inf : -Inf
