# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# RELIABILITY BRANCHING STRATEGY
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    ReliabilityBranching <: AbstractBranchingStrategy

Reliability branching (Achterberg et al., 2005, Algorithm 3).
Uses pseudocost estimates for reliable variables and CG probes
for unreliable ones. Candidates sorted by pseudocost score.
Lookahead `lookahead` stops probing when the best score
stabilizes.
"""
struct ReliabilityBranching <: AbstractBranchingStrategy
    max_candidates::Int
    max_cg_iterations::Int
    mu::Float64
    reliability_threshold::Int
    lookahead::Int
    # Any because orig_var type depends on DWReformulation{X}
    pseudocosts::PseudocostTracker{Any}

    function ReliabilityBranching(;
        max_candidates::Int = 100,
        max_cg_iterations::Int = 10,
        mu::Float64 = 1.0 / 6.0,
        reliability_threshold::Int = 8,
        lookahead::Int = 8
    )
        new(
            max_candidates, max_cg_iterations, mu,
            reliability_threshold, lookahead,
            PseudocostTracker{Any}(
                reliability_threshold=reliability_threshold
            )
        )
    end
end

function _rb_get_parent_lp(node)
    isnothing(node) && return nothing
    isnothing(node.user_data) && return nothing
    isnothing(node.user_data.cg_output) && return nothing
    return node.user_data.cg_output.master_lp_obj
end

function _rb_score_and_sort!(
    rb::ReliabilityBranching, candidates
)
    scored = [(c, estimate_score(rb.pseudocosts, c; mu=rb.mu))
              for c in candidates]
    sort!(scored; by=x -> x[2], rev=true)
    if length(scored) > rb.max_candidates
        resize!(scored, rb.max_candidates)
    end
    return scored
end

function _rb_evaluate_candidate(
    rb::ReliabilityBranching, space, c, pc_score,
    parent_lp::Float64
)
    if is_reliable(rb.pseudocosts, c)
        return pc_score, false, nothing
    end
    probe = run_sb_probe(
        space, c, rb.max_cg_iterations, parent_lp
    )
    if probe.left.is_infeasible && probe.right.is_infeasible
        return nothing, true, probe
    end
    update_pseudocosts!(rb.pseudocosts, c, probe)
    score = sb_score(probe; mu=rb.mu)
    return score, false, probe
end

function select_branching_variable(
    rb::ReliabilityBranching, space, node,
    primal_values::Dict{MOI.VariableIndex,Float64}
)
    ctx = space.ctx
    candidates = find_fractional_variables(
        ctx, primal_values; tol=space.tol
    )
    isempty(candidates) && return BranchingResult(all_integral)

    parent_lp = _rb_get_parent_lp(node)
    if isnothing(parent_lp)
        @warn "ReliabilityBranching: no parent LP obj, " *
              "falling back to most fractional candidate"
        c = first(candidates)
        return BranchingResult(c.orig_var, c.value)
    end

    log = space.log_level > 0
    t0 = time()
    log && _sb_log_header(stdout)

    scored = _rb_score_and_sort!(rb, candidates)

    best_score = -Inf
    best_candidate = scored[1][1]
    no_improvement_count = 0

    for (idx, (c, pc_score)) in enumerate(scored)
        score, both_inf, probe = _rb_evaluate_candidate(
            rb, space, c, pc_score, parent_lp
        )
        if both_inf
            log && println(stdout,
                "  RB cand. $(lpad(idx, 2)) branch on " *
                "$(c.orig_var): both infeasible"
            )
            return BranchingResult(node_infeasible)
        end
        if log && !isnothing(probe)
            _sb_log_candidate(
                stdout, idx, c, probe, score, t0
            )
        elseif log
            et = @sprintf("%.2f", time() - t0)
            lhs = @sprintf("%.4f", c.value)
            sc = @sprintf("%.2f", score)
            println(stdout,
                "  RB cand. $(lpad(idx, 2)) branch on " *
                "$(c.orig_var) (lhs=$(lhs)): " *
                "reliable, score = $(sc)  <et=$(et)>"
            )
        end

        if score > best_score
            best_score = score
            best_candidate = c
            no_improvement_count = 0
        else
            no_improvement_count += 1
        end

        if no_improvement_count >= rb.lookahead
            log && println(stdout,
                "  RB lookahead: no improvement for " *
                "$(no_improvement_count) candidates, stopping"
            )
            break
        end
    end

    log && _sb_log_selected(stdout, best_candidate, best_score)
    return BranchingResult(
        best_candidate.orig_var, best_candidate.value
    )
end

function on_node_evaluated(
    rb::ReliabilityBranching, space, node, cg_output
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
        rb.pseudocosts.records, bvar, PseudocostRecord()
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
