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
    run_branching_selection(space, node, phases, pseudocosts,
        primal_values; max_candidates, mu, tol, log_level)
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
    space, node,
    phases::Vector{<:AbstractBranchingPhase},
    pseudocosts::PseudocostTracker,
    primal_values::Dict{MOI.VariableIndex,Float64};
    max_candidates::Int = 100,
    mu::Float64 = 1.0 / 6.0,
    tol::Float64 = 1e-6,
    log_level::Int = 0
)
    ctx = space.ctx
    candidates = find_fractional_variables(
        ctx, primal_values; tol=tol
    )
    isempty(candidates) && return BranchingResult(all_integral)

    parent_lp = _get_parent_lp(node)
    if isnothing(parent_lp)
        @warn "run_branching_selection: no parent LP obj," *
              " falling back to most fractional candidate"
        c = first(candidates)
        return BranchingResult(c.orig_var, c.value)
    end

    current = select_initial_candidates(
        pseudocosts, candidates, max_candidates; mu=mu
    )

    log = log_level > 0
    t0 = time()
    log && _sb_log_header(stdout)

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
                phase, space, pseudocosts, c,
                parent_lp, idx, mu, log, t0
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
        if log && !isnothing(next_phase)
            println(stdout,
                "  [$(label)] filtered: " *
                "$(before) -> $(length(current)) candidates"
            )
        end
    end

    log && _sb_log_selected(stdout, best_candidate, best_score)
    return BranchingResult(
        best_candidate.orig_var, best_candidate.value
    )
end

# ─────────────────────────────────────────────────────────────
# CANDIDATE EVALUATION (internal)
# ─────────────────────────────────────────────────────────────

function _eval_candidate(
    phase::AbstractBranchingPhase, space,
    pseudocosts::PseudocostTracker,
    c::BranchingCandidate, parent_lp::Float64,
    idx::Int, mu::Float64, log::Bool, t0::Float64
)
    # Reliability skip for CG phases
    if phase isa CGProbePhase &&
       is_reliable(pseudocosts, c)
        score = estimate_score(pseudocosts, c; mu=mu)
        if log
            label = phase_label(phase)
            et = @sprintf("%.2f", time() - t0)
            lhs = @sprintf("%.4f", c.value)
            sc = @sprintf("%.2f", score)
            println(stdout,
                "  [$(label)] cand. $(lpad(idx, 2))" *
                " branch on $(c.orig_var)" *
                " (lhs=$(lhs)): reliable," *
                " score = $(sc)  <et=$(et)>"
            )
        end
        return score
    end

    result = probe_candidate(phase, space, c, parent_lp)

    if result.left.is_infeasible &&
       result.right.is_infeasible
        label = phase_label(phase)
        log && println(stdout,
            "  [$(label)] cand. $(lpad(idx, 2))" *
            " branch on $(c.orig_var):" *
            " both infeasible"
        )
        return nothing  # signals node_infeasible
    end

    score = score_candidate(phase, result; mu=mu)
    update_pseudocosts!(pseudocosts, c, result)

    if log
        label = phase_label(phase)
        _sb_log_candidate(
            stdout, idx, c, result, score, t0
        )
    end
    return score
end
