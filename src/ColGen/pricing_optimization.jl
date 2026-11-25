# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ── Pricing strategy ──────────────────────────────────────────────────────────

struct DefaultPricingStrategy{I}
    pricing_sps::I
end

function get_pricing_strategy(ws::ColGenWorkspace, ::CGPhase)
    return DefaultPricingStrategy(get_pricing_subprobs(ws))
end

pricing_strategy_iterate(s::DefaultPricingStrategy) = iterate(s.pricing_sps)
pricing_strategy_iterate(s::DefaultPricingStrategy, state) = iterate(s.pricing_sps, state)

# ── Pricing solution ──────────────────────────────────────────────────────────

"""
    PricingPrimalSolution

Primal solution returned by a single pricing subproblem.

Wraps the sparse variable–value pairs (`solution`) together with the
subproblem identity (`sp_id`). The value of `solution` is the subproblem
objective `(c - πA)ᵀz̄` alone, without the fixed cost `fₖ` and the
convexity duals ν. The framework adds them to get the reduced cost of the
column and decides whether it is improving; see [`push_in_set!`](@ref).
"""
struct PricingPrimalSolution
    sp_id::PricingSubproblemId
    solution::_SpSolution
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

A pricer returns raw subproblem values only: both bounds, and the value
of each column in `primal_sols`, are values of the subproblem objective
`(c - πA)ᵀz̄` alone. They exclude the fixed cost `fₖ` and the convexity
duals ν, which the framework accounts for itself, in
[`compute_dual_bound`](@ref) and [`push_in_set!`](@ref) (Sadykov,
*Modern Branch-Cut-and-Price*, 2019, §2.2.1). Only the framework knows
the phase, and in Phase 1 `fₖ` counts as 0. A pricer that works in
reduced costs, `fₖ` and ν included, removes them before filling the
bounds and the columns.

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
non-improving solutions. Every column in it has an improving reduced
cost.
"""
struct GeneratedColumns
    collection::Vector{PricingPrimalSolution}
end

function set_of_columns(::ColGenWorkspace)
    return GeneratedColumns(PricingPrimalSolution[])
end

"""
    _column_reduced_cost(ws, phase, sp_id, sp_value, mast_dual_sol) -> Float64

Reduced cost of a column of subproblem `sp_id` whose subproblem objective
`(c - πA)ᵀz̄` is `sp_value`: `sp_value + fₖ - ν_lb - ν_ub`. The fixed cost
`fₖ` follows the phase, like the cost of the column in the master.
"""
function _column_reduced_cost(
    ws::ColGenWorkspace, phase::CGPhase, sp_id, sp_value,
    mast_dual_sol::MasterDualSolution
)
    decomp = ws.decomp
    fk = _objective_cost(phase, subproblem_fixed_cost(decomp, sp_id))
    ν_lb = _get_convexity_dual(mast_dual_sol, convexity_lb_pairs(decomp), sp_id)
    ν_ub = _get_convexity_dual(mast_dual_sol, convexity_ub_pairs(decomp), sp_id)
    return sp_value + fk - ν_lb - ν_ub
end

# Minimization: improving when reduced_cost < -ε (min(rc) is very negative).
# Maximization: improving when reduced_cost > +ε (max(rc) is positive).
function _is_improving_reduced_cost(ws::ColGenWorkspace, reduced_cost)
    if is_minimization(ws)
        return reduced_cost < -RC_IMPROVING_TOL
    else
        return reduced_cost > RC_IMPROVING_TOL
    end
end

"""
    push_in_set!(ws, phase, set, sol, mast_dual_sol) -> Bool

Add the column `sol` to `set` if its reduced cost under `phase` is
improving, and return whether it was added. The reduced cost is computed
here, from the raw subproblem value the pricer returned.
"""
function push_in_set!(
    ws::ColGenWorkspace, phase::CGPhase, set::GeneratedColumns,
    sol::PricingPrimalSolution, mast_dual_sol::MasterDualSolution
)
    reduced_cost = _column_reduced_cost(
        ws, phase, sol.sp_id, sol.solution.obj_value, mast_dual_sol
    )
    if _is_improving_reduced_cost(ws, reduced_cost)
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

# ── Optional subproblem ───────────────────────────────────────────────────────

"""
    is_subproblem_optional(ws, sp_id) -> Bool

Return `true` when the lower multiplicity of subproblem `sp_id` is zero: the
master may leave it unused, so an infeasible pricing does not make the master
infeasible.
"""
function is_subproblem_optional(ws::ColGenWorkspace, sp_id)
    conv_lb, _ = convexity_bounds(ws.decomp, sp_id)
    return conv_lb <= 0
end

# ── optimize_pricing_problem! ─────────────────────────────────────────────────

"""
    optimize_pricing_problem!(ws, sp_id, pricing_sp, optimizer, mast_dual_sol, stab_changes_mast_dual_sol) -> PricingSolution

Solve the pricing subproblem `sp_id`, whose objective already holds the
reduced costs of its variables, and return what the optimizer found.

A pricer returns raw subproblem values only: the bounds and the value of
each column are values of `(c - πA)ᵀz̄`, without the fixed cost `fₖ` and
the convexity duals ν. The framework adds `fₖ`, which depends on the
phase, and ν; see [`PricingSolution`](@ref).
"""
function optimize_pricing_problem!(
    ws::ColGenWorkspace,
    sp_id,
    pricing_sp::PricingSubproblem,
    ::SubproblemMoiOptimizer,
    mast_dual_sol::MasterDualSolution,
    stab_changes_mast_dual_sol
)
    sp_model = moi_pricing_sp(pricing_sp)
    sp_sense = is_minimization(ws) ? MOI.MIN_SENSE : MOI.MAX_SENSE
    MOI.set(sp_model, MOI.ObjectiveSense(), sp_sense)
    MOI.optimize!(sp_model)

    status = MOI.get(sp_model, MOI.TerminationStatus())
    is_inf = status == MOI.INFEASIBLE
    is_unb = status == MOI.DUAL_INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED

    if is_inf || is_unb
        return PricingSolution(is_inf, is_unb, 0.0, 0.0, PricingPrimalSolution[])
    end

    sp_obj = MOI.get(sp_model, MOI.ObjectiveValue())

    # Extract solution — filter to SP decision variables
    sp_vars = Set(subproblem_variables(ws.decomp, sp_id))
    all_vals = get_primal_solution(sp_model)
    entries = Tuple{MOI.VariableIndex,Float64}[]
    for (var, val) in all_vals
        if var in sp_vars
            push!(entries, (var, val))
        end
    end
    sol = _SpSolution(sp_id, sp_obj, entries)

    primal_sol = PricingPrimalSolution(sp_id, sol)

    return PricingSolution(false, false, sp_obj, sp_obj, [primal_sol])
end
