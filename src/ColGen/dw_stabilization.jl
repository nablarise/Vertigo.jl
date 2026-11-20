# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ────────────────────────────────────────────────────────────────────────────────────────
# NO STABILIZATION (default)
# ────────────────────────────────────────────────────────────────────────────────────────

function update_stabilization_after_master_optim!(
    ::NoStabilization, phase, ::MasterDualSolution
)
    return false
end

get_stab_dual_sol(::NoStabilization, phase, dual_sol::MasterDualSolution) = dual_sol

function update_stabilization_after_pricing_optim!(
    ::NoStabilization, ::ColGenWorkspace, _, _, _, _
)
    return nothing
end

check_misprice(::NoStabilization, _, _) = false

update_stabilization_after_misprice!(::NoStabilization, _) = nothing

update_stabilization_after_iter!(::NoStabilization, ::MasterDualSolution) = nothing

# ────────────────────────────────────────────────────────────────────────────────────────
# WENTGES SMOOTHING
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    WentgesSmoothing

Dual smoothing stabilization for column generation (Wentges 1997).

Computes a smoothed dual solution `π^sep = α·π^in + (1-α)·π^out` that
lies between a stability center (π^in) and the LP dual (π^out), reducing
oscillation and accelerating convergence.
"""
mutable struct WentgesSmoothing
    ws::ColGenWorkspace
    smooth_dual_sol_coeff::Float64
    cur_smooth_dual_sol_coeff::Float64
    stab_center::Union{Nothing,MasterDualSolution}
    # π̂^τ, the center at the start of the iteration. Pricing may move
    # `stab_center` before the angle test, which measures from here.
    in_point::Union{Nothing,MasterDualSolution}
    best_lagrangian_bound::Float64
    nb_misprices::Int
    last_phase::Any
    # One pricing solution per subproblem, from the first separation point of
    # the iteration, π̃₀: the angle test reads the subgradient there (Sadykov
    # 2019, Step 6), not at the separation points of the mispricings.
    sep_dual_sol::Union{Nothing,MasterDualSolution}
    oracle_sols::Union{Nothing,Dict{PricingSubproblemId,_SpSolution}}
end

# ── Setup ────────────────────────────────────────────────────────────────────

function setup_stabilization!(ws::ColGenWorkspace, _master)
    if ws.smoothing_alpha > 0.0
        init_lb = is_minimization(ws) ? -Inf : Inf
        return WentgesSmoothing(
            ws, ws.smoothing_alpha, ws.smoothing_alpha,
            nothing, nothing, init_lb, 0, nothing, nothing, nothing
        )
    end
    return NoStabilization()
end

# ── After master optimization ────────────────────────────────────────────────

function update_stabilization_after_master_optim!(
    stab::WentgesSmoothing, phase, mast_dual_sol::MasterDualSolution
)
    stab.oracle_sols = nothing

    # Phase change: reset center and best bound
    if phase != stab.last_phase
        stab.stab_center = nothing
        stab.best_lagrangian_bound = is_minimization(stab.ws) ? -Inf : Inf
        stab.last_phase = phase
    end

    # First iteration with this phase: initialize center
    if isnothing(stab.stab_center)
        stab.stab_center = mast_dual_sol
        stab.in_point = mast_dual_sol
        return false
    end
    stab.in_point = stab.stab_center

    # Reset misprice state for the new iteration
    stab.cur_smooth_dual_sol_coeff = stab.smooth_dual_sol_coeff
    stab.nb_misprices = 0
    return stab.smooth_dual_sol_coeff > 0.0
end

# ── Get smoothed dual solution ───────────────────────────────────────────────

function _convex_combination(
    center::MasterDualSolution,
    out::MasterDualSolution,
    alpha::Float64
)
    one_minus_alpha = 1.0 - alpha
    d_in = center.sol.constraint_duals
    d_out = out.sol.constraint_duals

    combined = Dict{TaggedCI,Float64}()
    for (idx, val) in d_in
        out_val = get(d_out, idx, 0.0)
        combined[idx] = alpha * val + one_minus_alpha * out_val
    end
    for (idx, val) in d_out
        haskey(combined, idx) && continue
        combined[idx] = one_minus_alpha * val
    end

    obj = alpha * center.sol.obj_value +
        one_minus_alpha * out.sol.obj_value
    return MasterDualSolution(
        DualMoiSolution(obj, combined),
        out.coupling_constraint_ids
    )
end

function get_stab_dual_sol(
    stab::WentgesSmoothing, _phase, mast_dual_sol::MasterDualSolution
)
    if stab.cur_smooth_dual_sol_coeff <= 0.0 || isnothing(stab.stab_center)
        return mast_dual_sol
    end
    return _convex_combination(
        stab.stab_center, mast_dual_sol, stab.cur_smooth_dual_sol_coeff
    )
end

# ── After pricing optimization ───────────────────────────────────────────────

function update_stabilization_after_pricing_optim!(
    stab::WentgesSmoothing,
    ws::ColGenWorkspace,
    sp_primal_sols,
    _master,
    pseudo_db,
    sep_mast_dual_sol
)
    if isnothing(stab.oracle_sols)
        stab.sep_dual_sol = sep_mast_dual_sol
        # A pricer may return no solution, e.g. none below its threshold: that
        # subproblem is left out of the subgradient, which is exact when its
        # multiplicity is a lower convexity bound of zero.
        stab.oracle_sols = Dict(
            sp_id => _oracle_solution(ws, sols)
            for (sp_id, sols) in sp_primal_sols if !isempty(sols)
        )
    end

    if is_minimization(ws)
        if pseudo_db > stab.best_lagrangian_bound
            stab.best_lagrangian_bound = pseudo_db
            stab.stab_center = sep_mast_dual_sol
        end
    else
        if pseudo_db < stab.best_lagrangian_bound
            stab.best_lagrangian_bound = pseudo_db
            stab.stab_center = sep_mast_dual_sol
        end
    end
    return nothing
end

# ── Misprice check ───────────────────────────────────────────────────────────

function check_misprice(
    stab::WentgesSmoothing,
    generated_columns::GeneratedColumns,
    mast_dual_sol::MasterDualSolution
)
    stab.cur_smooth_dual_sol_coeff <= 0.0 && return false

    ws = stab.ws
    decomp = ws.decomp

    for pricing_sol in generated_columns.collection
        sp_id = pricing_sol.sp_id
        sol = pricing_sol.solution

        # Compute RC at π^out (the original LP dual, not the smoothed one)
        sp_rc_at_out = _compute_sp_reduced_costs(ws, mast_dual_sol, sp_id)

        ν_lb = _get_convexity_dual(mast_dual_sol, convexity_lb_pairs(decomp), sp_id)
        ν_ub = _get_convexity_dual(mast_dual_sol, convexity_ub_pairs(decomp), sp_id)

        rc = compute_column_reduced_cost(
            decomp, sp_id, sol, sp_rc_at_out, (ν_lb, ν_ub)
        )

        # If any column improves at π^out, no misprice
        is_imp = is_minimization(ws) ? (rc < -RC_IMPROVING_TOL) : (rc > RC_IMPROVING_TOL)
        is_imp && return false
    end

    return true
end

# ── After misprice ───────────────────────────────────────────────────────────

function update_stabilization_after_misprice!(
    stab::WentgesSmoothing, _mast_dual_sol
)
    stab.nb_misprices += 1
    if stab.nb_misprices >= stab.ws.max_misprice_iterations
        stab.cur_smooth_dual_sol_coeff = 0.0
    else
        stab.cur_smooth_dual_sol_coeff = max(
            0.0,
            1.0 - stab.nb_misprices * (1.0 - stab.smooth_dual_sol_coeff)
        )
    end
    return nothing
end

# ── After iteration (auto-adjust α) ─────────────────────────────────────────

function update_stabilization_after_iter!(
    stab::WentgesSmoothing, mast_dual_sol::MasterDualSolution
)
    isnothing(stab.in_point) && return nothing
    isnothing(stab.oracle_sols) && return nothing

    g = _lagrangian_subgradient(
        stab.ws, stab.last_phase, stab.sep_dual_sol, stab.oracle_sols
    )
    direction_product = 0.0
    for (cstr_id, g_i) in g
        pi_out = _dual_value(mast_dual_sol, cstr_id)
        pi_in = _dual_value(stab.in_point, cstr_id)
        direction_product += g_i * (pi_out - pi_in)
    end

    # Adjust α based on direction product
    if direction_product > 0
        stab.smooth_dual_sol_coeff = max(
            0.0, stab.smooth_dual_sol_coeff - 0.1
        )
    else
        stab.smooth_dual_sol_coeff += (
            1.0 - stab.smooth_dual_sol_coeff
        ) * 0.1
    end

    return nothing
end

# The pricing optimum among the solutions a subproblem returned.
function _oracle_solution(ws::ColGenWorkspace, pricing_sols)
    sense = is_minimization(ws) ? 1 : -1
    return argmin(ps -> sense * ps.solution.obj_value, pricing_sols).solution
end

# Value of an oracle solution in the Lagrangian function at `dual_sol`:
# fₖ + c̄z̄. The reduced cost the pricing reported is not it: it also subtracts
# the convexity duals.
function _oracle_value(ws::ColGenWorkspace, ::Union{Phase0,Phase2}, dual_sol, sp_id, sol)
    sp_rc = _compute_sp_reduced_costs(ws, dual_sol, sp_id)
    return compute_column_reduced_cost(ws.decomp, sp_id, sol, sp_rc, (0.0, 0.0))
end

# Phase 1 sets every column's cost to zero, its fixed cost included.
function _oracle_value(ws::ColGenWorkspace, ::Phase1, dual_sol, sp_id, sol)
    sp_rc = _compute_sp_reduced_costs(ws, dual_sol, sp_id; zero_cost=true)
    return sum(sp_rc[sp_var] * val for (sp_var, val) in nonzero_entries(sol); init=0.0)
end

# Subgradient of the Lagrangian function at `sep_dual_sol`, where the
# subproblems returned `oracle_sols`, on the dualized rows: coupling
# constraints, robust cuts and branching constraints.
#     g_i = rhs_i - Σ_k m_k Σ_t A_{i,t}^k z_t^k - Σ_s φ_{i,s} ỹ_s
#                                                     (Sadykov 2019, eq. 2.29)
# m_k is the multiplicity the dual bound gives subproblem k, ỹ_s the bound it
# gives pure master variable s. A column's coefficient in a cut or branching
# row comes from its original variables, as in `_insert_column!`.
function _lagrangian_subgradient(ws::ColGenWorkspace, phase, sep_dual_sol, oracle_sols)
    decomp = ws.decomp
    master = master_model(decomp)
    master_rhs(tagged) = with_typed_ci(
        ci -> _rhs(MOI.get(master, MOI.ConstraintSet(), ci)), tagged
    )

    g = Dict{TaggedCI,Float64}()
    for (cstr_id, rhs) in coupling_constraints(decomp)
        g[cstr_id] = rhs
    end
    for cut in ws.robust_cuts
        g[cut.constraint_index] = master_rhs(cut.constraint_index)
    end
    for bc in ws.branching_constraints
        g[bc.constraint_index] = master_rhs(bc.constraint_index)
    end

    for (sp_id, sol) in oracle_sols
        sp_value = _oracle_value(ws, phase, sep_dual_sol, sp_id, sol)
        mult = _lagrangian_multiplicity(decomp, sp_id, sp_value)
        for (sp_var, val) in nonzero_entries(sol)
            for entry in coupling_coefficients(decomp, sp_id, sp_var)
                g[entry.constraint_id] -= mult * entry.coefficient * val
            end
            ov = mapped_original_var(decomp, sp_id, sp_var)
            if ov !== nothing
                for cut in ws.robust_cuts
                    coeff = get(cut.coefficients, ov, 0.0)
                    g[cut.constraint_index] -= mult * coeff * val
                end
                for bc in ws.branching_constraints
                    if bc.orig_var == ov
                        g[bc.constraint_index] -= mult * val
                    end
                end
            end
        end
    end
    _subtract_pure_master_activity!(g, ws, phase, sep_dual_sol)
    return g
end

# Pure master variables appear in coupling constraints only.
function _subtract_pure_master_activity!(g, ws::ColGenWorkspace, phase, sep_dual_sol)
    decomp = ws.decomp
    for y_var in pure_master_variables(decomp)
        coeffs = pure_master_coupling_coefficients(decomp, y_var)
        rc_y = _pure_master_cost(decomp, y_var, phase)
        for entry in coeffs
            rc_y -= entry.coefficient * _dual_value(sep_dual_sol, entry.constraint_id)
        end
        y = pure_master_lagrangian_value(decomp, y_var, rc_y)
        for entry in coeffs
            g[entry.constraint_id] -= entry.coefficient * y
        end
    end
    return g
end

_pure_master_cost(decomp, y_var, ::Union{Phase0,Phase2}) = pure_master_cost(decomp, y_var)
# Phase 1 sets the cost of pure master variables to zero.
_pure_master_cost(_decomp, _y_var, ::Phase1) = 0.0

# ── Iteration logging dispatch ───────────────────────────────────────────────

function after_colgen_iteration(
    ::ColGenWorkspace,
    ::CGPhase,
    ::ExactStage,
    ::Int64,
    ::WentgesSmoothing,
    ::ColGenIterationOutput,
    _incumbent_dual_bound,
    _ip_primal_sol
)
    return nothing
end
