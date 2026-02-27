# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Pricing strategy ──────────────────────────────────────────────────────────

struct DefaultPricingStrategy{I}
    pricing_sps::I
end

function get_pricing_strategy(ctx::ColGenContext, ::Union{Phase0,Phase1,Phase2})
    return DefaultPricingStrategy(get_pricing_subprobs(ctx))
end

pricing_strategy_iterate(s::DefaultPricingStrategy) = iterate(s.pricing_sps)
pricing_strategy_iterate(s::DefaultPricingStrategy, state) = iterate(s.pricing_sps, state)

# ── Pricing solution ──────────────────────────────────────────────────────────

struct PricingSolution{PS}
    is_infeasible::Bool
    is_unbounded::Bool
    primal_bound::Float64
    dual_bound::Float64
    primal_sols::Vector{PS}
end

is_infeasible(sol::PricingSolution) = sol.is_infeasible
is_unbounded(sol::PricingSolution) = sol.is_unbounded
get_primal_sols(sol::PricingSolution) = sol.primal_sols
get_primal_bound(sol::PricingSolution) = sol.primal_bound
get_dual_bound(sol::PricingSolution) = sol.dual_bound

struct PricingPrimalSolution{S,V}
    sp_id::S
    solution::SpSolution{S,V}
    is_improving::Bool
end

# ── Set of columns ────────────────────────────────────────────────────────────

struct GeneratedColumns
    collection::Vector{Any}
end

set_of_columns(::ColGenContext) = GeneratedColumns(Any[])

function push_in_set!(set::GeneratedColumns, sol::PricingPrimalSolution)
    if sol.is_improving
        push!(set.collection, sol)
        return true
    end
    return false
end

# ── Pricing subproblem optimizer ──────────────────────────────────────────────

struct SubproblemMoiOptimizer end

get_pricing_subprob_optimizer(::ExactStage, ::PricingSubproblem) = SubproblemMoiOptimizer()

# ── Convexity dual helper ─────────────────────────────────────────────────────

function _get_convexity_dual(mast_dual_sol::MasterDualSolution, convexity_dict, sp_id)
    haskey(convexity_dict, sp_id) || return 0.0
    return _dual_value(mast_dual_sol, convexity_dict[sp_id])
end

# ── optimize_pricing_problem! ─────────────────────────────────────────────────

function optimize_pricing_problem!(
    ctx::ColGenContext,
    sp_id,
    pricing_sp::PricingSubproblem,
    ::SubproblemMoiOptimizer,
    mast_dual_sol::MasterDualSolution,
    stab_changes_mast_dual_sol
)
    sp_model = moi_pricing_sp(pricing_sp)
    MOI.optimize!(sp_model)

    status = MOI.get(sp_model, MOI.TerminationStatus())
    is_inf = status == MOI.INFEASIBLE
    is_unb = status == MOI.DUAL_INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED

    if is_inf || is_unb
        return PricingSolution(is_inf, is_unb, 0.0, 0.0, PricingPrimalSolution[])
    end

    sp_obj = MOI.get(sp_model, MOI.ObjectiveValue())

    ν_lb = _get_convexity_dual(mast_dual_sol, ctx.convexity_lb, sp_id)
    ν_ub = _get_convexity_dual(mast_dual_sol, ctx.convexity_ub, sp_id)
    fk = subproblem_fixed_cost(ctx.decomp, sp_id)
    reduced_cost = sp_obj + fk - ν_lb - ν_ub

    is_improving = if is_minimization(ctx)
        reduced_cost < -1e-6
    else
        reduced_cost > 1e-6
    end

    # Extract solution — only iterate SP decision variables from decomp
    entries = Tuple{MOI.VariableIndex,Float64}[]
    for sp_var in subproblem_variables(ctx.decomp, sp_id)
        val = MOI.get(sp_model, MOI.VariablePrimal(), sp_var)
        if abs(val) > 1e-8
            push!(entries, (sp_var, val))
        end
    end
    sol = SpSolution(sp_id, reduced_cost, entries)

    primal_sol = PricingPrimalSolution(sp_id, sol, is_improving)

    return PricingSolution(false, false, reduced_cost, sp_obj, [primal_sol])
end
