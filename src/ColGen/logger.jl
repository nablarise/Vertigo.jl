# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ──────────────────────────────────────────────────────────────────────
# VERTIGO LOGGER WORKSPACE
# A thin wrapper around ColGenWorkspace that overrides the two logging
# hooks to emit compact, tag-based terminal output.
# ──────────────────────────────────────────────────────────────────────

mutable struct ColGenLoggerWorkspace
    inner::ColGenWorkspace
    cg_start_time::Float64
    log_level::Int
    log_frequency::Int
    master_time::Float64
    pricing_time::Float64
    function ColGenLoggerWorkspace(
        ws::ColGenWorkspace; log_level::Int=1, log_frequency::Int=1
    )
        return new(ws, 0.0, log_level, log_frequency, 0.0, 0.0)
    end
end

# ── Formatting helpers ───────────────────────────────────────────────

_alpha_val(::NoStabilization) = 0.0
_alpha_val(s::WentgesSmoothing) = s.smooth_dual_sol_coeff

function _fmt_bound(x)
    if isnothing(x)
        return "N/A"
    end
    if isinf(x)
        return x < 0 ? "-Inf" : "Inf"
    end
    return @sprintf("%.4f", x)
end

# ── Protocol delegation (workspace as arg 1) ─────────────────────────

new_phase_iterator(lws::ColGenLoggerWorkspace) = new_phase_iterator(lws.inner)
new_stage_iterator(lws::ColGenLoggerWorkspace) = new_stage_iterator(lws.inner)
function setup_stabilization!(lws::ColGenLoggerWorkspace, master)
    return setup_stabilization!(lws.inner, master)
end
stop_colgen(lws::ColGenLoggerWorkspace, args...) = stop_colgen(lws.inner, args...)
colgen_output_type(lws::ColGenLoggerWorkspace) = colgen_output_type(lws.inner)
new_output(O, lws::ColGenLoggerWorkspace, args...) = new_output(O, lws.inner, args...)
function stop_colgen_phase(lws::ColGenLoggerWorkspace, args...)
    return stop_colgen_phase(lws.inner, args...)
end
function is_better_dual_bound(lws::ColGenLoggerWorkspace, args...)
    return is_better_dual_bound(lws.inner, args...)
end
colgen_phase_output_type(lws::ColGenLoggerWorkspace) = colgen_phase_output_type(lws.inner)
function new_phase_output(O, lws::ColGenLoggerWorkspace, args...)
    return new_phase_output(O, lws.inner, args...)
end
is_minimization(lws::ColGenLoggerWorkspace) = is_minimization(lws.inner)
get_master(lws::ColGenLoggerWorkspace) = get_master(lws.inner)
get_reform(lws::ColGenLoggerWorkspace) = get_reform(lws.inner)
function colgen_iteration_output_type(lws::ColGenLoggerWorkspace)
    return colgen_iteration_output_type(lws.inner)
end
function update_inc_primal_sol!(lws::ColGenLoggerWorkspace, args...)
    return update_inc_primal_sol!(lws.inner, args...)
end
function update_master_constrs_dual_vals!(lws::ColGenLoggerWorkspace, args...)
    return update_master_constrs_dual_vals!(lws.inner, args...)
end
function compute_reduced_costs!(lws::ColGenLoggerWorkspace, args...)
    return compute_reduced_costs!(lws.inner, args...)
end
function update_reduced_costs!(lws::ColGenLoggerWorkspace, args...)
    return update_reduced_costs!(lws.inner, args...)
end
set_of_columns(lws::ColGenLoggerWorkspace) = set_of_columns(lws.inner)
function get_pricing_strategy(lws::ColGenLoggerWorkspace, args...)
    return get_pricing_strategy(lws.inner, args...)
end
get_pricing_subprobs(lws::ColGenLoggerWorkspace) = get_pricing_subprobs(lws.inner)
insert_columns!(lws::ColGenLoggerWorkspace, args...) = insert_columns!(lws.inner, args...)
function compute_dual_bound(lws::ColGenLoggerWorkspace, args...)
    return compute_dual_bound(lws.inner, args...)
end
compute_sp_init_db(lws::ColGenLoggerWorkspace, sp) = compute_sp_init_db(lws.inner, sp)
compute_sp_init_pb(lws::ColGenLoggerWorkspace, sp) = compute_sp_init_pb(lws.inner, sp)
max_cg_iterations(lws::ColGenLoggerWorkspace) = max_cg_iterations(lws.inner)
function set_max_cg_iterations!(lws::ColGenLoggerWorkspace, n::Int)
    return set_max_cg_iterations!(lws.inner, n)
end

# ── Protocol delegation (workspace as arg 2) ─────────────────────────

function update_stabilization_after_pricing_optim!(
    stab, lws::ColGenLoggerWorkspace, args...
)
    return update_stabilization_after_pricing_optim!(stab, lws.inner, args...)
end

function check_primal_ip_feasibility!(sol, lws::ColGenLoggerWorkspace, phase)
    return check_primal_ip_feasibility!(sol, lws.inner, phase)
end

# ── Timing wrappers ──────────────────────────────────────────────────

function optimize_master_lp_problem!(master, lws::ColGenLoggerWorkspace)
    t0 = time()
    result = optimize_master_lp_problem!(master, lws.inner)
    lws.master_time += time() - t0
    return result
end

function optimize_pricing_problem!(lws::ColGenLoggerWorkspace, args...)
    t0 = time()
    result = optimize_pricing_problem!(lws.inner, args...)
    lws.pricing_time += time() - t0
    return result
end

# ── Logging overrides ────────────────────────────────────────────────

function setup_reformulation!(lws::ColGenLoggerWorkspace, phase)
    lws.master_time = 0.0
    lws.pricing_time = 0.0
    return setup_reformulation!(lws.inner, phase)
end

function after_colgen_iteration(
    lws::ColGenLoggerWorkspace,
    phase,
    _stage,
    colgen_iterations,
    stab,
    out,
    incumbent_dual_bound,
    _ip_primal_sol,
)
    if lws.log_level == 0
        return nothing
    end

    mst = lws.master_time
    spt = lws.pricing_time
    lws.master_time = 0.0
    lws.pricing_time = 0.0

    is_terminal = out.nb_columns_added == 0
    if !is_terminal && colgen_iterations % lws.log_frequency != 0
        return nothing
    end

    prefix = if phase isa Phase0
        "  "
    elseif phase isa Phase1
        "# "
    else
        "##"
    end

    et = time() - lws.cg_start_time
    al = _alpha_val(stab)
    db_star = _fmt_bound(incumbent_dual_bound)
    mlp = _fmt_bound(out.master_lp_obj)
    ip_val = isnothing(lws.inner.ip_incumbent) ? nothing : lws.inner.ip_incumbent.obj_value
    pb = _fmt_bound(ip_val)

    line = string(
        prefix,
        @sprintf(
            "<it=%3d> <et=%5.2f> <cols=%2d> <al=%5.2f>",
            colgen_iterations,
            et,
            out.nb_columns_added,
            al
        ),
        " <DB*=",
        db_star,
        "> <mlp=",
        mlp,
        "> <PB=",
        pb,
        ">",
    )

    if lws.log_level >= 2
        db = _fmt_bound(out.dual_bound)
        line *= string(@sprintf(" <mst=%5.2f> <spt=%5.2f>", mst, spt), " <DB=", db, ">")
    end

    return println(line)
end

# ── Entry point ──────────────────────────────────────────────────────

"""
    run_column_generation(lws::ColGenLoggerWorkspace) -> ColGenOutput

Run column generation with compact, tag-based terminal logging.

# Examples
```jldoctest
julia> # (see test for a full example)
```
"""
function run_column_generation(lws::ColGenLoggerWorkspace)
    lws.cg_start_time = time()
    output = ColGen.run!(lws, nothing)
    _print_cg_footer(lws, output)
    return output
end

function _print_cg_footer(lws::ColGenLoggerWorkspace, output::ColGenOutput)
    if lws.log_level == 0
        return nothing
    end
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
    if !isnothing(lws.inner.ip_incumbent)
        @printf("[STATUS] IP incumbent: %.6e\n", lws.inner.ip_incumbent.obj_value)
    end
end
