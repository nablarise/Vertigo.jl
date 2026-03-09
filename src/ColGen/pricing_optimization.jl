# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Pricing strategy ──────────────────────────────────────────────────────────

struct DefaultPricingStrategy{I}
    pricing_sps::I
end

function get_pricing_strategy(ctx::ColGenContext, ::CGPhase)
    return DefaultPricingStrategy(get_pricing_subprobs(ctx))
end

pricing_strategy_iterate(s::DefaultPricingStrategy) = iterate(s.pricing_sps)
pricing_strategy_iterate(s::DefaultPricingStrategy, state) = iterate(s.pricing_sps, state)

# ── Pricing solution ──────────────────────────────────────────────────────────

"""
    PricingPrimalSolution

Primal solution returned by a single pricing subproblem.

Wraps the sparse variable–value pairs (`solution`) together with the
subproblem identity (`sp_id`) and a flag indicating whether the
reduced cost is improving (negative for minimization, positive for
maximization).
"""
struct PricingPrimalSolution
    sp_id::PricingSubproblemId
    solution::_SpSolution
    is_improving::Bool
end

"""
    PricingSolution

Raw optimizer output for a single pricing subproblem call.

Captures termination information (`is_infeasible`, `is_unbounded`),
objective bounds, and every primal solution found by the optimizer.
A `PricingSolution` is transient: it is produced by
[`optimize_pricing_problem!`](@ref), inspected once, and discarded.

## Bound semantics

The two bounds correspond to the standard primal/dual pair of the
**subproblem optimization** (not the master):

- `primal_bound`: objective value of the best *feasible* solution found.
- `dual_bound`: best known bound on the subproblem *optimum* (e.g., from
  an internal LP relaxation or branch-and-bound).

When the subproblem is solved to optimality (exact pricing),
both values are equal.  The distinction matters for **heuristic pricing**,
where the solver returns a feasible solution without proving optimality.

These bounds are used differently at the master level:

- `dual_bound` → feeds the *valid* master dual bound via
  [`compute_dual_bound`](@ref). This bound is tight and drives the
  convergence test (LP gap closed?).
- `primal_bound` → feeds the *pseudo* dual bound, which is weaker
  (uses feasible SP values instead of proven optima). The pseudo bound
  is used only by stabilization: a looser bound avoids deactivating
  smoothing prematurely after a heuristic pricing round.

See also [`GeneratedColumns`](@ref), which accumulates only the
*improving* solutions across all subproblems within one column
generation iteration.
"""
struct PricingSolution
    is_infeasible::Bool
    is_unbounded::Bool
    primal_bound::Float64
    dual_bound::Float64
    primal_sols::Vector{PricingPrimalSolution}
end

is_infeasible(sol::PricingSolution) = sol.is_infeasible
is_unbounded(sol::PricingSolution) = sol.is_unbounded
get_primal_sols(sol::PricingSolution) = sol.primal_sols
get_primal_bound(sol::PricingSolution) = sol.primal_bound
get_dual_bound(sol::PricingSolution) = sol.dual_bound

# ── Set of columns ────────────────────────────────────────────────────────────

"""
    GeneratedColumns

Accumulator for improving columns across all subproblems within one
column generation iteration.

Unlike [`PricingSolution`](@ref), which is the raw output of a single
subproblem call (including infeasibility flags and bounds),
`GeneratedColumns` retains only the solutions whose reduced cost is
improving. These are the columns that will be inserted into the
restricted master problem.

Columns are added via [`push_in_set!`](@ref), which silently discards
non-improving solutions.
"""
struct GeneratedColumns
    collection::Vector{PricingPrimalSolution}
end

function set_of_columns(::ColGenContext)
    return GeneratedColumns(PricingPrimalSolution[])
end

function push_in_set!(set::GeneratedColumns, sol::PricingPrimalSolution)
    if sol.is_improving
        push!(set.collection, sol)
        return true
    end
    return false
end

# ── Pricing subproblem optimizer ──────────────────────────────────────────────

"""
    SubproblemMoiOptimizer

Exact pricing optimizer that solves the subproblem via `MOI.optimize!`.

Selected by [`get_pricing_subprob_optimizer`](@ref) for the
[`ExactStage`](@ref) pricing stage.
"""
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
    sp_sense = is_minimization(ctx) ? MOI.MIN_SENSE : MOI.MAX_SENSE
    MOI.set(sp_model, MOI.ObjectiveSense(), sp_sense)
    MOI.optimize!(sp_model)

    status = MOI.get(sp_model, MOI.TerminationStatus())
    is_inf = status == MOI.INFEASIBLE
    is_unb = status == MOI.DUAL_INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED

    if is_inf || is_unb
        return PricingSolution(is_inf, is_unb, 0.0, 0.0, PricingPrimalSolution[])
    end

    sp_obj = MOI.get(sp_model, MOI.ObjectiveValue())

    ν_lb = _get_convexity_dual(mast_dual_sol, convexity_lb_pairs(ctx.decomp), sp_id)
    ν_ub = _get_convexity_dual(mast_dual_sol, convexity_ub_pairs(ctx.decomp), sp_id)
    fk = subproblem_fixed_cost(ctx.decomp, sp_id)
    reduced_cost = sp_obj + fk - ν_lb - ν_ub

    # Minimization: improving when reduced_cost < -ε (min(rc) is very negative).
    # Maximization: improving when reduced_cost > +ε (max(rc) is positive).
    is_improving = if is_minimization(ctx)
        reduced_cost < -RC_IMPROVING_TOL
    else
        reduced_cost > RC_IMPROVING_TOL
    end

    # Extract solution — only iterate SP decision variables from decomp
    entries = Tuple{MOI.VariableIndex,Float64}[]
    for sp_var in subproblem_variables(ctx.decomp, sp_id)
        val = MOI.get(sp_model, MOI.VariablePrimal(), sp_var)
        if abs(val) > SOLUTION_ENTRY_TOL
            push!(entries, (sp_var, val))
        end
    end
    sol = _SpSolution(sp_id, reduced_cost, entries)

    primal_sol = PricingPrimalSolution(sp_id, sol, is_improving)

    return PricingSolution(false, false, reduced_cost, sp_obj, [primal_sol])
end
