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

function _capture_probe_state(ctx, space)
    return (
        max_iter = ColGen.max_cg_iterations(ctx),
        ip_inc = bp_ip_incumbent(ctx),
        ip_bound = bp_ip_primal_bound(ctx),
        bcs = copy(bp_branching_constraints(ctx)),
        basis = MathOptState.capture_basis(space.backend),
    )
end

function _restore_probe_state!(ctx, space, snapshot)
    bcs = bp_branching_constraints(ctx)
    empty!(bcs)
    append!(bcs, snapshot.bcs)
    ColGen.set_max_cg_iterations!(ctx, snapshot.max_iter)
    bp_set_ip_primal_bound!(ctx, snapshot.ip_bound)
    bp_set_ip_incumbent!(ctx, snapshot.ip_inc)
    MathOptState.apply_change!(
        space.backend,
        MathOptState.LPBasisDiff(snapshot.basis),
        nothing
    )
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
    space::BPSpace, candidate::BranchingCandidate,
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
# STRONG BRANCHING STRATEGY
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    StrongBranching <: AbstractBranchingStrategy

Evaluate candidate variables with limited CG probes and pick
the one with the best product score.
"""
struct StrongBranching <: AbstractBranchingStrategy
    max_candidates::Int
    max_cg_iterations::Int
    mu::Float64
    rule::AbstractBranchingRule

    function StrongBranching(;
        max_candidates::Int = 5,
        max_cg_iterations::Int = 10,
        mu::Float64 = 1.0 / 6.0,
        rule::AbstractBranchingRule = MostFractionalRule()
    )
        new(max_candidates, max_cg_iterations, mu, rule)
    end
end

function select_branching_variable(
    sb::StrongBranching, space, node,
    primal_values::Dict{MOI.VariableIndex,Float64}
)
    ctx = space.ctx
    candidates = find_fractional_variables(
        ctx, primal_values; tol=space.tol
    )
    isempty(candidates) && return BranchingResult(all_integral)

    selected = select_candidates(
        sb.rule, candidates, sb.max_candidates
    )

    parent_lp = if !isnothing(node) &&
                   !isnothing(node.user_data) &&
                   !isnothing(node.user_data.cg_output)
        node.user_data.cg_output.master_lp_obj
    else
        nothing
    end

    if isnothing(parent_lp)
        @warn "StrongBranching: no parent LP obj available, " *
              "falling back to most fractional candidate"
        c = first(selected)
        return BranchingResult(c.orig_var, c.value)
    end

    best_score = -Inf
    best_candidate = first(selected)

    for c in selected
        probe = run_sb_probe(
            space, c, sb.max_cg_iterations, parent_lp
        )
        if probe.left.is_infeasible && probe.right.is_infeasible
            @debug "SB: both children infeasible" var=c.orig_var
            return BranchingResult(node_infeasible)
        end
        score = sb_score(probe; mu=sb.mu)
        @debug "SB candidate scored" var=c.orig_var score=score
        if score > best_score
            best_score = score
            best_candidate = c
        end
    end
    @debug "SB selected" var=best_candidate.orig_var score=best_score
    return BranchingResult(best_candidate.orig_var, best_candidate.value)
end
