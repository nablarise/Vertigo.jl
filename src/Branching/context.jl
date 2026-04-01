# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    BranchingContext

Abstract type for branching contexts. All hook functions
dispatch on this type — default implementations are no-ops.
"""
abstract type BranchingContext end

"""
    DefaultBranchingContext <: BranchingContext

Default branching context with no-op hooks.
"""
struct DefaultBranchingContext <: BranchingContext end

# ── Hook stubs (no-ops) ─────────────────────────────────────

function before_branching_selection(
    ::BranchingContext, candidates, phases
)
    return
end

function before_probe(
    ::BranchingContext, phase, candidate, direction::Symbol
)
    return
end

function after_probe(
    ::BranchingContext, phase, candidate,
    direction::Symbol, result
)
    return
end

function after_candidate_eval(
    ::BranchingContext, phase, idx::Int, candidate,
    score::Float64, detail
)
    return
end

function on_both_infeasible(
    ::BranchingContext, phase, idx::Int, candidate
)
    return
end

function after_phase_filter(
    ::BranchingContext, label::String,
    before::Int, after::Int
)
    return
end

function after_branching_selection(
    ::BranchingContext, candidate, score::Float64
)
    return
end
