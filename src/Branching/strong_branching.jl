# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# PROBE RESULT TYPES
# ────────────────────────────────────────────────────────────────────────────────────────

struct SBProbeResult
    dual_bound::Union{Nothing,Float64}
    lp_obj::Union{Nothing,Float64}
    is_infeasible::Bool
end

struct SBCandidateResult{X}
    candidate::BranchingCandidate{X}
    parent_lp_obj::Float64
    left::SBProbeResult
    right::SBProbeResult
end

# ────────────────────────────────────────────────────────────────────────────────────────
# SCORING
# ────────────────────────────────────────────────────────────────────────────────────────

function _sb_delta(probe::SBProbeResult, parent_lp_obj::Float64)
    probe.is_infeasible && return Inf
    if isnothing(probe.dual_bound)
        @warn "SB probe returned no dual bound; scoring as Δ=0"
        return 0.0
    end
    return max(0.0, probe.dual_bound - parent_lp_obj)
end

"""
    sb_score(result; mu=1.0/6.0) -> Float64

Product score: `(1-μ) * min(Δ⁻, Δ⁺) + μ * max(Δ⁻, Δ⁺)`.
Infeasible child → Δ = Inf.
"""
function sb_score(result::SBCandidateResult; mu::Float64=1.0/6.0)
    d_left = _sb_delta(result.left, result.parent_lp_obj)
    d_right = _sb_delta(result.right, result.parent_lp_obj)
    return (1 - mu) * min(d_left, d_right) +
           mu * max(d_left, d_right)
end

# ────────────────────────────────────────────────────────────────────────────────────────
# BRANCHING CONSTRAINT HELPERS
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    build_branching_terms(decomp, pool, orig_var)

Build MOI constraint terms for a branching constraint on `orig_var`.
"""
function build_branching_terms(decomp, pool, orig_var)
    terms = MOI.ScalarAffineTerm{Float64}[]
    for (col_var, rec) in columns(pool)
        coeff = compute_branching_column_coefficient(
            decomp, orig_var, column_sp_id(rec), rec.solution
        )
        if !iszero(coeff)
            push!(terms, MOI.ScalarAffineTerm(coeff, col_var))
        end
    end
    return terms
end

"""
    add_branching_constraint!(backend, ctx, terms, set, orig_var)

Add a branching constraint to the MOI backend and register it
in `branching_constraints` in a single function call.
Returns the MOI constraint index.
"""
function add_branching_constraint!(backend, ctx, terms, set, orig_var)
    f = MOI.ScalarAffineFunction(terms, 0.0)
    ci = MOI.add_constraint(backend, f, set)
    bcs = bp_branching_constraints(ctx)
    push!(bcs, ColGen.ActiveBranchingConstraint(
        TaggedCI(ci), orig_var
    ))
    return ci
end

"""
    remove_branching_constraint!(backend, ctx, ci)

Delete the MOI constraint and remove it from `branching_constraints`.
Defensive: handles partial state gracefully.
"""
function remove_branching_constraint!(backend, ctx, ci)
    if MOI.is_valid(backend, ci)
        MOI.delete(backend, ci)
    end
    bcs = bp_branching_constraints(ctx)
    tagged = TaggedCI(ci)
    filter!(bc -> bc.constraint_index != tagged, bcs)
    return
end

# ────────────────────────────────────────────────────────────────────────────────────────
# PROBE EXECUTION
# ────────────────────────────────────────────────────────────────────────────────────────

function _try_capture_basis(backend)
    try
        return MathOptState.capture_basis(backend)
    catch
        return nothing
    end
end

function _capture_probe_state(ctx, space)
    return (
        max_iter = ColGen.max_cg_iterations(ctx),
        ip_inc = bp_ip_incumbent(ctx),
        ip_bound = bp_ip_primal_bound(ctx),
        bcs = copy(bp_branching_constraints(ctx)),
        basis = _try_capture_basis(space.backend),
    )
end

function _restore_probe_state!(ctx, space, snapshot)
    bcs = bp_branching_constraints(ctx)
    empty!(bcs)
    append!(bcs, snapshot.bcs)
    ColGen.set_max_cg_iterations!(ctx, snapshot.max_iter)
    bp_set_ip_primal_bound!(ctx, snapshot.ip_bound)
    bp_set_ip_incumbent!(ctx, snapshot.ip_inc)
    if !isnothing(snapshot.basis)
        try
            MathOptState.apply_change!(
                space.backend,
                MathOptState.LPBasisDiff(snapshot.basis),
                nothing
            )
        catch
            # Basis restore failed (e.g., variable mismatch
            # after Phase 0/2 artificial variable lifecycle).
            # Fall through — solver will re-solve from scratch.
        end
    end
    return
end

function _run_one_direction(space, candidate, set, max_cg_iter)
    ctx = space.ctx
    backend = space.backend
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)

    terms = build_branching_terms(decomp, pool, candidate.orig_var)
    ci = add_branching_constraint!(
        backend, ctx, terms, set, candidate.orig_var
    )
    ColGen.set_max_cg_iterations!(ctx, max_cg_iter)

    try
        cg_output = ColGen.run_column_generation(ctx)
        is_inf = cg_output.status == ColGen.master_infeasible ||
                 cg_output.status == ColGen.subproblem_infeasible
        return SBProbeResult(
            cg_output.incumbent_dual_bound,
            cg_output.master_lp_obj,
            is_inf
        )
    finally
        remove_branching_constraint!(backend, ctx, ci)
    end
end

"""
    run_sb_probe(space, candidate, max_cg_iterations, parent_lp_obj)

Run strong branching probes in both directions (floor/ceil) for
the given candidate. Captures and restores context state (iteration
limit, IP incumbent, primal bound, branching constraints, LP basis)
around both probes. Returns `SBCandidateResult`.
"""
function run_sb_probe(
    space, candidate::BranchingCandidate,
    max_cg_iterations::Int, parent_lp_obj::Float64
)
    snapshot = _capture_probe_state(space.ctx, space)
    try
        left = _run_one_direction(
            space, candidate,
            MOI.LessThan(candidate.floor_val),
            max_cg_iterations
        )
        _restore_probe_state!(space.ctx, space, snapshot)
        right = _run_one_direction(
            space, candidate,
            MOI.GreaterThan(candidate.ceil_val),
            max_cg_iterations
        )
        @debug "SB probe" candidate.orig_var left right
        return SBCandidateResult(
            candidate, parent_lp_obj, left, right
        )
    finally
        _restore_probe_state!(space.ctx, space, snapshot)
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# STRONG BRANCHING LOGGING
# ────────────────────────────────────────────────────────────────────────────────────────

function _sb_log_header(io::IO)
    println(io, "**** Strong branching ****")
    return
end

function _sb_fmt_bound(probe::SBProbeResult)
    probe.is_infeasible && return "infeasible"
    isnothing(probe.dual_bound) && return "N/A"
    return @sprintf("%.4f", probe.dual_bound)
end

function _sb_log_candidate(
    io::IO, idx::Int, candidate::BranchingCandidate,
    result::SBCandidateResult, score::Float64, t0::Float64
)
    lhs = @sprintf("%.4f", candidate.value)
    left_str = _sb_fmt_bound(result.left)
    right_str = _sb_fmt_bound(result.right)
    et = @sprintf("%.2f", time() - t0)
    sc = @sprintf("%.2f", score)
    println(io,
        "  SB cand. $(lpad(idx, 2)) branch on " *
        "$(candidate.orig_var) (lhs=$(lhs)): " *
        "[$(left_str), $(right_str)], " *
        "score = $(sc)  <et=$(et)>"
    )
    return
end

function _sb_log_selected(
    io::IO, candidate::BranchingCandidate, score::Float64
)
    sc = @sprintf("%.2f", score)
    println(io,
        "  SB selected: $(candidate.orig_var) (score = $(sc))"
    )
    return
end

