# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ──────────────────────────────────────────────────────────────────────
# VERTIGO LOGGER CONTEXT
# A thin wrapper around ColGenContext that overrides the two logging
# hooks to emit compact, tag-based terminal output.
# ──────────────────────────────────────────────────────────────────────

mutable struct ColGenLoggerContext
    inner::ColGenContext
    cg_start_time::Float64
    log_level::Int
    log_frequency::Int
    master_time::Float64
    pricing_time::Float64
    function ColGenLoggerContext(
        ctx::ColGenContext;
        log_level::Int=1,
        log_frequency::Int=1
    )
        return new(ctx, 0.0, log_level, log_frequency, 0.0, 0.0)
    end
end

# ── Formatting helpers ───────────────────────────────────────────────

_alpha_val(::NoStabilization) = 0.0
_alpha_val(s::WentgesSmoothing) = s.smooth_dual_sol_coeff

function _fmt_bound(x)
    isnothing(x) && return "N/A"
    isinf(x) && return x < 0 ? "-Inf" : "Inf"
    return @sprintf("%.4f", x)
end

# ── Protocol delegation (context as arg 1) ───────────────────────────

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
compute_sp_init_db(lctx::ColGenLoggerContext, sp)                    = compute_sp_init_db(lctx.inner, sp)
compute_sp_init_pb(lctx::ColGenLoggerContext, sp)                    = compute_sp_init_pb(lctx.inner, sp)
max_cg_iterations(lctx::ColGenLoggerContext)                         = max_cg_iterations(lctx.inner)
set_max_cg_iterations!(lctx::ColGenLoggerContext, n::Int)            = set_max_cg_iterations!(lctx.inner, n)

# ── Protocol delegation (context as arg 2) ───────────────────────────

update_stabilization_after_pricing_optim!(stab, lctx::ColGenLoggerContext, args...) =
    update_stabilization_after_pricing_optim!(stab, lctx.inner, args...)

check_primal_ip_feasibility!(sol, lctx::ColGenLoggerContext, phase) =
    check_primal_ip_feasibility!(sol, lctx.inner, phase)

# ── Timing wrappers ──────────────────────────────────────────────────

function optimize_master_lp_problem!(master, lctx::ColGenLoggerContext)
    t0 = time()
    result = optimize_master_lp_problem!(master, lctx.inner)
    lctx.master_time += time() - t0
    return result
end

function optimize_pricing_problem!(lctx::ColGenLoggerContext, args...)
    t0 = time()
    result = optimize_pricing_problem!(lctx.inner, args...)
    lctx.pricing_time += time() - t0
    return result
end

# ── Logging overrides ────────────────────────────────────────────────

function setup_reformulation!(lctx::ColGenLoggerContext, phase)
    lctx.master_time = 0.0
    lctx.pricing_time = 0.0
    setup_reformulation!(lctx.inner, phase)
end

function after_colgen_iteration(
    lctx::ColGenLoggerContext, phase, _stage,
    colgen_iterations, stab, out,
    incumbent_dual_bound, _ip_primal_sol
)
    lctx.log_level == 0 && return

    mst = lctx.master_time
    spt = lctx.pricing_time
    lctx.master_time = 0.0
    lctx.pricing_time = 0.0

    is_terminal = out.nb_columns_added == 0
    if !is_terminal &&
        colgen_iterations % lctx.log_frequency != 0
        return
    end

    prefix = phase isa Phase0 ? "  " :
             phase isa Phase1 ? "# " : "##"

    et = time() - lctx.cg_start_time
    al = _alpha_val(stab)
    db_star = _fmt_bound(incumbent_dual_bound)
    mlp = _fmt_bound(out.master_lp_obj)
    ip_val = isnothing(lctx.inner.ip_incumbent) ? nothing :
             lctx.inner.ip_incumbent.obj_value
    pb = _fmt_bound(ip_val)

    line = string(
        prefix,
        @sprintf("<it=%3d> <et=%5.2f> <cols=%2d> <al=%5.2f>",
                 colgen_iterations, et, out.nb_columns_added, al),
        " <DB*=", db_star,
        "> <mlp=", mlp,
        "> <PB=", pb, ">"
    )

    if lctx.log_level >= 2
        db = _fmt_bound(out.dual_bound)
        line *= string(
            @sprintf(" <mst=%5.2f> <spt=%5.2f>", mst, spt),
            " <DB=", db, ">"
        )
    end

    println(line)
end

# ── Entry point ──────────────────────────────────────────────────────

"""
    run_column_generation(lctx::ColGenLoggerContext) -> ColGenOutput

Run column generation with compact, tag-based terminal logging.

# Examples
```jldoctest
julia> # (see test for a full example)
```
"""
function run_column_generation(lctx::ColGenLoggerContext)
    lctx.cg_start_time = time()
    output = ColGen.run!(lctx, nothing)
    _print_cg_footer(lctx, output)
    return output
end

function _print_cg_footer(
    lctx::ColGenLoggerContext, output::ColGenOutput
)
    lctx.log_level == 0 && return
    println()
    if output.status == optimal
        println("[STATUS] Convergence reached.")
    elseif output.status == master_infeasible
        println("[STATUS] Infeasible master problem.")
    elseif output.status == subproblem_infeasible
        println("[STATUS] Subproblem infeasible.")
    elseif output.status == ip_pruned
        println("[STATUS] Node pruned by bound.")
    else
        println("[STATUS] Iteration limit reached.")
    end
    if !isnothing(lctx.inner.ip_incumbent)
        @printf(
            "[STATUS] IP incumbent: %.6e\n",
            lctx.inner.ip_incumbent.obj_value
        )
    end
end
