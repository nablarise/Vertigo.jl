# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ──────────────────────────────────────────────────────────────
# BRANCHING LOGGER CONTEXT
# A thin wrapper around BranchingContext that overrides hook
# functions to emit formatted terminal output.
# ──────────────────────────────────────────────────────────────

mutable struct BranchingLoggerContext <: BranchingContext
    io::IO
    log_level::Int
    t0::Float64

    function BranchingLoggerContext(;
        io::IO = stdout,
        log_level::Int = 1
    )
        return new(io, log_level, 0.0)
    end
end

# ── Formatting helpers ──────────────────────────────────────

function _sb_fmt_bound(probe::SBProbeResult)
    probe.is_infeasible && return "infeasible"
    isnothing(probe.dual_bound) && return "N/A"
    return @sprintf("%.4f", probe.dual_bound)
end

# ── Hook overrides ──────────────────────────────────────────

function before_branching_selection(
    lctx::BranchingLoggerContext, candidates, phases
)
    lctx.t0 = time()
    println(lctx.io, "**** Strong branching ****")
    return
end

function before_probe(
    lctx::BranchingLoggerContext,
    phase::AbstractBranchingPhase,
    candidate, direction::Symbol
)
    lctx.log_level < 2 && return
    label = phase_label(phase)
    println(lctx.io,
        "  [$(label)] probing $(candidate.orig_var)" *
        " direction=$(direction)"
    )
    return
end

function after_probe(
    lctx::BranchingLoggerContext,
    phase::AbstractBranchingPhase,
    candidate, direction::Symbol, result
)
    lctx.log_level < 2 && return
    label = phase_label(phase)
    bound_str = _sb_fmt_bound(result)
    println(lctx.io,
        "  [$(label)] probe $(direction)" *
        " $(candidate.orig_var): $(bound_str)"
    )
    return
end

function after_reliability_skip(
    lctx::BranchingLoggerContext, phase, idx::Int,
    candidate, score::Float64
)
    label = phase_label(phase)
    et = @sprintf("%.2f", time() - lctx.t0)
    lhs = @sprintf("%.4f", candidate.value)
    sc = @sprintf("%.2f", score)
    println(lctx.io,
        "  [$(label)] cand. $(lpad(idx, 2))" *
        " branch on $(candidate.orig_var)" *
        " (lhs=$(lhs)): reliable," *
        " score = $(sc)  <et=$(et)>"
    )
    return
end

function after_candidate_probed(
    lctx::BranchingLoggerContext, phase, idx::Int,
    candidate, score::Float64, result::SBCandidateResult
)
    label = phase_label(phase)
    et = @sprintf("%.2f", time() - lctx.t0)
    lhs = @sprintf("%.4f", candidate.value)
    sc = @sprintf("%.2f", score)
    left_str = _sb_fmt_bound(result.left)
    right_str = _sb_fmt_bound(result.right)
    println(lctx.io,
        "  [$(label)] cand. $(lpad(idx, 2)) branch on " *
        "$(candidate.orig_var) (lhs=$(lhs)): " *
        "[$(left_str), $(right_str)], " *
        "score = $(sc)  <et=$(et)>"
    )
    return
end

function on_both_infeasible(
    lctx::BranchingLoggerContext, phase, idx::Int, candidate
)
    label = phase_label(phase)
    println(lctx.io,
        "  [$(label)] cand. $(lpad(idx, 2))" *
        " branch on $(candidate.orig_var):" *
        " both infeasible"
    )
    return
end

function after_phase_filter(
    lctx::BranchingLoggerContext, label::String,
    before::Int, after::Int
)
    println(lctx.io,
        "  [$(label)] filtered: " *
        "$(before) -> $(after) candidates"
    )
    return
end

function after_branching_selection(
    lctx::BranchingLoggerContext, candidate, score::Float64
)
    sc = @sprintf("%.2f", score)
    println(lctx.io,
        "  SB selected: $(candidate.orig_var)" *
        " (score = $(sc))"
    )
    return
end
