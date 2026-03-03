# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# VERTIGO LOGGER CONTEXT
# A thin wrapper around ColGenContext that overrides the two logging hooks to emit
# styled, aligned terminal output without touching ColGenContext itself.
# ────────────────────────────────────────────────────────────────────────────────────────

mutable struct ColGenLoggerContext
    inner::ColGenContext
    cg_start_time::Float64
    header_printed::Bool
    ColGenLoggerContext(ctx::ColGenContext) = new(ctx, 0.0, false)
end

# ── Table formatting ──────────────────────────────────────────────────────────

const _VRT_HDR = "    ITER        OBJ (LP)       BEST DUAL        DUAL BND       IP PRIMAL   COLS   TIME (s)"
const _VRT_SEP = "  ------   -------------   -------------   -------------   -------------   ----   --------"
const _VRT_HDR_ALPHA = "    ALPHA"
const _VRT_SEP_ALPHA = "   ------"

_fmt_alpha(::NoStabilization) = ""
_fmt_alpha(s::WentgesSmoothing) = @sprintf("   %6.4f", s.smooth_dual_sol_coeff)

_fmt_val(::Nothing) = "          N/A"
function _fmt_val(x::Float64)
    isinf(x) && return x < 0 ? "         -inf" : "          inf"
    return @sprintf("%13.6e", x)
end

# ── Protocol delegation (context as arg 1) ───────────────────────────────────

new_phase_iterator(lctx::ColGenLoggerContext)                        = new_phase_iterator(lctx.inner)
new_stage_iterator(lctx::ColGenLoggerContext)                        = new_stage_iterator(lctx.inner)
setup_stabilization!(lctx::ColGenLoggerContext, master)              = setup_stabilization!(lctx.inner, master)
stop_colgen(lctx::ColGenLoggerContext, args...)                      = stop_colgen(lctx.inner, args...)
colgen_output_type(lctx::ColGenLoggerContext)                        = colgen_output_type(lctx.inner)
new_output(O, lctx::ColGenLoggerContext, args...)                    = new_output(O, lctx.inner, args...)
stop_colgen_phase(lctx::ColGenLoggerContext, args...)                = stop_colgen_phase(lctx.inner, args...)
is_better_dual_bound(lctx::ColGenLoggerContext, args...)             = is_better_dual_bound(lctx.inner, args...)
colgen_phase_output_type(lctx::ColGenLoggerContext)                  = colgen_phase_output_type(lctx.inner)
new_phase_output(O, lctx::ColGenLoggerContext, args...)              = new_phase_output(O, lctx.inner, args...)
is_minimization(lctx::ColGenLoggerContext)                           = is_minimization(lctx.inner)
get_master(lctx::ColGenLoggerContext)                                = get_master(lctx.inner)
get_reform(lctx::ColGenLoggerContext)                                = get_reform(lctx.inner)
colgen_iteration_output_type(lctx::ColGenLoggerContext)              = colgen_iteration_output_type(lctx.inner)
update_inc_primal_sol!(lctx::ColGenLoggerContext, args...)           = update_inc_primal_sol!(lctx.inner, args...)
update_master_constrs_dual_vals!(lctx::ColGenLoggerContext, args...) = update_master_constrs_dual_vals!(lctx.inner, args...)
compute_reduced_costs!(lctx::ColGenLoggerContext, args...)           = compute_reduced_costs!(lctx.inner, args...)
update_reduced_costs!(lctx::ColGenLoggerContext, args...)            = update_reduced_costs!(lctx.inner, args...)
set_of_columns(lctx::ColGenLoggerContext)                            = set_of_columns(lctx.inner)
get_pricing_strategy(lctx::ColGenLoggerContext, args...)             = get_pricing_strategy(lctx.inner, args...)
get_pricing_subprobs(lctx::ColGenLoggerContext)                      = get_pricing_subprobs(lctx.inner)
insert_columns!(lctx::ColGenLoggerContext, args...)                  = insert_columns!(lctx.inner, args...)
compute_dual_bound(lctx::ColGenLoggerContext, args...)               = compute_dual_bound(lctx.inner, args...)
optimize_pricing_problem!(lctx::ColGenLoggerContext, args...)        = optimize_pricing_problem!(lctx.inner, args...)
compute_sp_init_db(lctx::ColGenLoggerContext, sp)                    = compute_sp_init_db(lctx.inner, sp)
compute_sp_init_pb(lctx::ColGenLoggerContext, sp)                    = compute_sp_init_pb(lctx.inner, sp)

# ── Protocol delegation (context as arg 2) ───────────────────────────────────

optimize_master_lp_problem!(master, lctx::ColGenLoggerContext) =
    optimize_master_lp_problem!(master, lctx.inner)

update_stabilization_after_pricing_optim!(stab, lctx::ColGenLoggerContext, args...) =
    update_stabilization_after_pricing_optim!(stab, lctx.inner, args...)

check_primal_ip_feasibility!(sol, lctx::ColGenLoggerContext, phase) =
    check_primal_ip_feasibility!(sol, lctx.inner, phase)

# ── Logging overrides ─────────────────────────────────────────────────────────

# Print a blank line between phases; the header stays up (printed once at first iteration).
function setup_reformulation!(lctx::ColGenLoggerContext, phase)
    lctx.header_printed && println()
    setup_reformulation!(lctx.inner, phase)
end

function after_colgen_iteration(
    lctx::ColGenLoggerContext, phase, _stage,
    colgen_iterations, stab, out, incumbent_dual_bound, _ip_primal_sol
)
    if !lctx.header_printed
        has_alpha = lctx.inner.smoothing_alpha > 0.0
        println(_VRT_HDR * (has_alpha ? _VRT_HDR_ALPHA : ""))
        println(_VRT_SEP * (has_alpha ? _VRT_SEP_ALPHA : ""))
        lctx.header_printed = true
    end
    iter_tag = phase isa Phase0 ? "" : phase isa Phase1 ? "#" : "##"
    iter_str = @sprintf("%6s", iter_tag * string(colgen_iterations))
    lp     = out.master_lp_obj
    db     = incumbent_dual_bound
    db2    = out.dual_bound
    ip_val = isnothing(lctx.inner.ip_incumbent) ? nothing :
             lctx.inner.ip_incumbent.obj_value
    t      = time() - lctx.cg_start_time
    @printf "  %s   %s   %s   %s   %s   %4d   %8.2f" iter_str _fmt_val(lp) _fmt_val(db) _fmt_val(db2) _fmt_val(ip_val) out.nb_columns_added t
    println(_fmt_alpha(stab))
end

# ── Entry point ───────────────────────────────────────────────────────────────

"""
    run_column_generation(lctx::ColGenLoggerContext) -> ColGenOutput

Run column generation with VERTIGO-styled terminal logging.

# Examples
```jldoctest
julia> # (see test for a full example)
```
"""
function run_column_generation(lctx::ColGenLoggerContext)
    lctx.cg_start_time = time()
    println("[VERTIGO] :: Initializing Master Problem...")
    output = ColGen.run!(lctx, nothing)
    _print_cg_footer(lctx, output)
    return output
end

function _print_cg_footer(lctx::ColGenLoggerContext, output::ColGenOutput)
    println()
    if output.status == optimal
        println("[STATUS] :: Convergence reached. LP gap < 1e-6.")
        println("[SIGNAL] :: Optimal LP relaxation found.")
    elseif output.status == master_infeasible
        println("[STATUS] :: Infeasible master problem.")
        println("[SIGNAL] :: No feasible solution exists.")
    elseif output.status == subproblem_infeasible
        println("[STATUS] :: Subproblem infeasible.")
        println("[SIGNAL] :: Pricing failed — check decomposition.")
    else
        println("[STATUS] :: Iteration limit reached.")
        println("[SIGNAL] :: Terminating with current best solution.")
    end
    if !isnothing(lctx.inner.ip_incumbent)
        @printf "[SIGNAL] :: IP incumbent: %.6e\n" lctx.inner.ip_incumbent.obj_value
    end
end
