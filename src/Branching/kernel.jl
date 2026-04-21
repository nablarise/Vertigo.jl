# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ─────────────────────────────────────────────────────────────
# GENERIC MULTI-PHASE STRONG BRANCHING KERNEL
# ─────────────────────────────────────────────────────────────

"""
    select_initial_candidates(pseudocosts, candidates,
        max_candidates; mu=1/6) -> Vector{BranchingCandidate}

Sort `candidates` by pseudocost estimate descending and
truncate to `max_candidates`.
"""
function select_initial_candidates(
    pseudocosts::PseudocostTracker,
    candidates::Vector{<:BranchingCandidate},
    max_candidates::Int;
    mu::Float64 = 1.0 / 6.0
)
    scored = sort(
        candidates;
        by = c -> estimate_score(pseudocosts, c; mu=mu),
        rev = true
    )
    n = min(length(scored), max_candidates)
    return scored[1:n]
end

"""
    score_candidate(phase, result; mu=1/6) -> Float64

Score a probed candidate. Default: delegates to `sb_score`.
"""
function score_candidate(
    ::AbstractBranchingPhase,
    result::SBCandidateResult;
    mu::Float64 = 1.0 / 6.0
)
    return sb_score(result; mu=mu)
end

"""
    filter_candidates(phase, next_phase, scored)
        -> Vector{BranchingCandidate}

Keep the top `ceil(length * keep_fraction)` candidates from
`scored` (sorted descending by score). If `next_phase` is
`nothing` (last phase), keep all candidates.
"""
function filter_candidates(
    phase::AbstractBranchingPhase,
    next_phase,
    scored::Vector{<:Tuple{BranchingCandidate,Float64}}
)
    isnothing(next_phase) && return [s[1] for s in scored]
    n = ceil(Int, length(scored) * phase.keep_fraction)
    n = max(n, 1)
    return [scored[i][1] for i in 1:min(n, length(scored))]
end

"""
    stop_phase(phase, idx, best_score, no_improvement_count)
        -> Bool

Return `true` when the lookahead limit is reached. A phase
with `lookahead == 0` never triggers early stopping.
"""
function stop_phase(
    phase::AbstractBranchingPhase,
    idx::Int,
    best_score::Float64,
    no_improvement_count::Int
)
    phase.lookahead <= 0 && return false
    return no_improvement_count >= phase.lookahead
end

"""
    _get_parent_lp(node) -> Union{Nothing,Float64}

Extract the parent LP objective from a B&B node, returning
`nothing` if any intermediate field is missing.
"""
function _get_parent_lp(node)
    isnothing(node) && return nothing
    isnothing(node.user_data) && return nothing
    isnothing(node.user_data.cg_output) && return nothing
    return node.user_data.cg_output.master_lp_obj
end

# ─────────────────────────────────────────────────────────────
# MAIN KERNEL
# ─────────────────────────────────────────────────────────────

"""
    run_branching_selection(bctx, space, node, phases,
        pseudocosts, primal_values; max_candidates, mu, tol)
        -> BranchingResult

Generic multi-phase strong branching kernel.

1. Find fractional variables and select initial candidates
   ranked by pseudocost estimate.
2. For each phase, probe (or reliability-skip) each candidate,
   score it, and filter the surviving candidates down for the
   next phase.
3. Return the best candidate from the last phase.
"""
function run_branching_selection(
    bctx::BranchingContext, space, node,
    phases::Vector{<:AbstractBranchingPhase},
    pseudocosts::PseudocostTracker,
    primal_values::Dict{MOI.VariableIndex,Float64};
    max_candidates::Int = 100,
    mu::Float64 = 1.0 / 6.0,
    tol::Float64 = 1e-6
)
    ws = space.ws
    candidates = find_fractional_variables(
        ws, primal_values; tol=tol
    )
    isempty(candidates) && return BranchingResult(all_integral)

    parent_lp = _get_parent_lp(node)
    if isnothing(parent_lp)
        c = first(candidates)
        return BranchingResult(c.orig_var, c.value)
    end

    current = select_initial_candidates(
        pseudocosts, candidates, max_candidates; mu=mu
    )

    before_branching_selection(bctx, current, phases)

    best_candidate = first(current)
    best_score = -Inf

    for (phase_idx, phase) in enumerate(phases)
        label = phase_label(phase)
        next_phase = phase_idx < length(phases) ?
            phases[phase_idx + 1] : nothing

        scored = Tuple{BranchingCandidate,Float64}[]
        no_improvement_count = 0

        for (idx, c) in enumerate(current)
            if stop_phase(phase, idx, best_score,
                          no_improvement_count)
                break
            end

            score = _eval_candidate(
                bctx, phase, space, pseudocosts, c,
                parent_lp, idx, mu
            )
            isnothing(score) && return BranchingResult(
                node_infeasible
            )

            push!(scored, (c, score))
            if score > best_score
                best_score = score
                best_candidate = c
                no_improvement_count = 0
            else
                no_improvement_count += 1
            end
        end

        sort!(scored; by=x -> x[2], rev=true)

        before = length(scored)
        current = filter_candidates(phase, next_phase, scored)
        if !isnothing(next_phase)
            after_phase_filter(
                bctx, label, before, length(current)
            )
        end
    end

    after_branching_selection(bctx, best_candidate, best_score)
    return BranchingResult(
        best_candidate.orig_var, best_candidate.value
    )
end

# ─────────────────────────────────────────────────────────────
# CANDIDATE EVALUATION (internal)
# ─────────────────────────────────────────────────────────────

function _eval_candidate(
    bctx::BranchingContext,
    phase::AbstractBranchingPhase, space,
    pseudocosts::PseudocostTracker,
    c::BranchingCandidate, parent_lp::Float64,
    idx::Int, mu::Float64
)
    if phase isa CGProbePhase &&
       is_reliable(pseudocosts, c)
        score = estimate_score(pseudocosts, c; mu=mu)
        after_reliability_skip(
            bctx, phase, idx, c, score
        )
        return score
    end

    result = probe_candidate(bctx, phase, space, c, parent_lp)

    if result.left.is_infeasible &&
       result.right.is_infeasible
        on_both_infeasible(bctx, phase, idx, c)
        return nothing
    end

    score = score_candidate(phase, result; mu=mu)
    update_pseudocosts!(pseudocosts, c, result)

    after_candidate_probed(
        bctx, phase, idx, c, score, result
    )
    return score
end

# ─────────────────────────────────────────────────────────────
# MULTI-PHASE STRONG BRANCHING STRATEGY
# ─────────────────────────────────────────────────────────────

"""
    MultiPhaseStrongBranching <: AbstractBranchingStrategy

Multi-phase strong branching with configurable phases and
pseudocost-based candidate selection with reliability skip.
"""
struct MultiPhaseStrongBranching{X} <: AbstractBranchingStrategy
    max_candidates::Int
    mu::Float64
    phases::Vector{AbstractBranchingPhase}
    pseudocosts::PseudocostTracker{X}
    branching_ctx::BranchingContext

    function MultiPhaseStrongBranching{X}(;
        max_candidates::Int = 20,
        mu::Float64 = 1.0 / 6.0,
        phases::Vector{<:AbstractBranchingPhase} = AbstractBranchingPhase[
            LPProbePhase(keep_fraction=0.25),
            CGProbePhase(max_cg_iterations=10, lookahead=8)
        ],
        reliability_threshold::Int = 8,
        branching_ctx::BranchingContext = DefaultBranchingContext()
    ) where {X}
        new{X}(
            max_candidates, mu,
            convert(Vector{AbstractBranchingPhase}, phases),
            PseudocostTracker{X}(
                reliability_threshold=reliability_threshold
            ),
            branching_ctx
        )
    end
end

"""
    MultiPhaseStrongBranching(; kwargs...)

Convenience constructor that defaults the key type to `Any`.
Use `MultiPhaseStrongBranching{X}(; kwargs...)` when the key
type is known at construction time.
"""
function MultiPhaseStrongBranching(; kwargs...)
    return MultiPhaseStrongBranching{Any}(; kwargs...)
end

"""
    select_branching_variable(mpsb, space, node, primal_values)
        -> BranchingResult

Delegate to the generic multi-phase kernel using the strategy's
configured phases and pseudocost tracker.
"""
function select_branching_variable(
    mpsb::MultiPhaseStrongBranching, space, node,
    primal_values::Dict{MOI.VariableIndex,Float64}
)
    return run_branching_selection(
        mpsb.branching_ctx, space, node,
        mpsb.phases, mpsb.pseudocosts,
        primal_values;
        max_candidates=mpsb.max_candidates,
        mu=mpsb.mu,
        tol=space.tol
    )
end

"""
    on_node_evaluated(mpsb, space, node, cg_output)

Passive pseudocost update after CG completes on a node. Uses
the branching metadata stored in `node.user_data` to update
the pseudocost record for the variable that was branched on.
"""
function on_node_evaluated(
    mpsb::MultiPhaseStrongBranching, space, node, cg_output
)
    bvar = node.user_data.branching_var
    isnothing(bvar) && return
    parent_lp = node.user_data.parent_lp_obj
    isnothing(parent_lp) && return
    isnothing(cg_output.incumbent_dual_bound) && return
    dir = node.user_data.branching_direction
    isnothing(dir) && return
    frac = node.user_data.branching_frac
    isnothing(frac) && return

    delta = max(
        0.0, cg_output.incumbent_dual_bound - parent_lp
    )
    rec = get!(
        mpsb.pseudocosts.records, bvar, PseudocostRecord()
    )

    if dir == branch_down && frac > 0.0
        rec.sum_down += delta / frac
        rec.count_down += 1
    elseif dir == branch_up && (1.0 - frac) > 0.0
        rec.sum_up += delta / (1.0 - frac)
        rec.count_up += 1
    end
    return
end
