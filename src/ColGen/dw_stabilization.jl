# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

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
    ::NoStabilization, ::ColGenContext, _, _, _, _
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
    ctx::ColGenContext
    smooth_dual_sol_coeff::Float64
    cur_smooth_dual_sol_coeff::Float64
    stab_center::Union{Nothing,MasterDualSolution}
    best_lagrangian_bound::Float64
    nb_misprices::Int
    last_phase::Any
    last_generated_columns::Union{Nothing,GeneratedColumns}
    last_sep_dual_sol::Union{Nothing,MasterDualSolution}
end

# ── Setup ────────────────────────────────────────────────────────────────────

function setup_stabilization!(ctx::ColGenContext, _master)
    if ctx.smoothing_alpha > 0.0
        init_lb = is_minimization(ctx) ? -Inf : Inf
        return WentgesSmoothing(
            ctx, ctx.smoothing_alpha, ctx.smoothing_alpha,
            nothing, init_lb, 0, nothing, nothing, nothing
        )
    end
    return NoStabilization()
end

# ── After master optimization ────────────────────────────────────────────────

function update_stabilization_after_master_optim!(
    stab::WentgesSmoothing, phase, mast_dual_sol::MasterDualSolution
)
    # Phase change: reset center and best bound
    if phase != stab.last_phase
        stab.stab_center = nothing
        stab.best_lagrangian_bound = is_minimization(stab.ctx) ? -Inf : Inf
        stab.last_phase = phase
    end

    # First iteration with this phase: initialize center
    if isnothing(stab.stab_center)
        stab.stab_center = mast_dual_sol
        return false
    end

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
    ctx::ColGenContext,
    generated_columns,
    _master,
    pseudo_db,
    sep_mast_dual_sol
)
    stab.last_generated_columns = generated_columns
    stab.last_sep_dual_sol = sep_mast_dual_sol

    if is_minimization(ctx)
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

    ctx = stab.ctx
    decomp = ctx.decomp

    for pricing_sol in generated_columns.collection
        sp_id = pricing_sol.sp_id
        sol = pricing_sol.solution

        # Compute RC at π^out (the original LP dual, not the smoothed one)
        sp_rc_at_out = _compute_sp_reduced_costs(ctx, mast_dual_sol, sp_id)

        ν_lb = _get_convexity_dual(mast_dual_sol, convexity_lb_pairs(decomp), sp_id)
        ν_ub = _get_convexity_dual(mast_dual_sol, convexity_ub_pairs(decomp), sp_id)

        rc = compute_column_reduced_cost(
            decomp, sp_id, sol, sp_rc_at_out, (ν_lb, ν_ub)
        )

        # If any column improves at π^out, no misprice
        is_imp = is_minimization(ctx) ? (rc < -RC_IMPROVING_TOL) : (rc > RC_IMPROVING_TOL)
        is_imp && return false
    end

    return true
end

# ── After misprice ───────────────────────────────────────────────────────────

function update_stabilization_after_misprice!(
    stab::WentgesSmoothing, _mast_dual_sol
)
    stab.nb_misprices += 1
    stab.cur_smooth_dual_sol_coeff = max(
        0.0, 1.0 - stab.nb_misprices * (1.0 - stab.smooth_dual_sol_coeff)
    )
    return nothing
end

# ── After iteration (auto-adjust α) ─────────────────────────────────────────

function update_stabilization_after_iter!(
    stab::WentgesSmoothing, mast_dual_sol::MasterDualSolution
)
    isnothing(stab.stab_center) && return nothing
    isnothing(stab.last_generated_columns) && return nothing
    isnothing(stab.last_sep_dual_sol) && return nothing

    ctx = stab.ctx
    decomp = ctx.decomp
    center = stab.stab_center
    out = mast_dual_sol

    # Compute subgradient direction product:
    # Σ_i g^sep_i * (π^out_i - π^in_i)
    # where g^sep_i = rhs_i - Σ_k m_k * Σ_t A_{i,t}^k * z_t^k
    direction_product = 0.0

    # Compute column contributions per coupling constraint
    col_contrib = Dict{Any,Float64}()
    for pricing_sol in stab.last_generated_columns.collection
        sp_id = pricing_sol.sp_id
        sol = pricing_sol.solution

        # Multiplicity: use convexity bound based on RC sign
        sense = is_minimization(ctx) ? 1 : -1
        rc = sol.obj_value
        conv_lb, conv_ub = convexity_bounds(decomp, sp_id)
        mult = (sense * rc < 0) ? conv_ub : conv_lb

        for (sp_var, val) in nonzero_entries(sol)
            for entry in coupling_coefficients(decomp, sp_id, sp_var)
                cstr = entry.constraint_id
                col_contrib[cstr] = get(col_contrib, cstr, 0.0) +
                    entry.coefficient * val * mult
            end
        end
    end

    # Iterate coupling constraints to compute direction product
    for (cstr_id, rhs) in coupling_constraints(decomp)
        g_sep = rhs - get(col_contrib, cstr_id, 0.0)
        pi_out = _dual_value(out, cstr_id)
        pi_in = _dual_value(center, cstr_id)
        direction_product += g_sep * (pi_out - pi_in)
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

# ── Iteration logging dispatch ───────────────────────────────────────────────

function after_colgen_iteration(
    ::ColGenContext,
    ::Union{Phase0,Phase1,Phase2},
    ::ExactStage,
    ::Int64,
    ::WentgesSmoothing,
    ::ColGenIterationOutput,
    _incumbent_dual_bound,
    _ip_primal_sol
)
    return nothing
end
