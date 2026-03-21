# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Probe result types ───────────────────────────────────────────────

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

# ── Scoring ──────────────────────────────────────────────────────────

function _sb_delta(probe::SBProbeResult, parent_lp_obj::Float64)
    probe.is_infeasible && return Inf
    isnothing(probe.dual_bound) && return 0.0
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

# ── Branching constraint helpers ─────────────────────────────────────

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
