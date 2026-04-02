# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    BranchingContext

Abstract type for branching contexts. Subtype to override
any of the following hooks (all default to no-ops):

- `before_branching_selection(ctx, candidates, phases)`
- `before_probe(ctx, phase, candidate, direction)`
- `after_probe(ctx, phase, candidate, direction, result)`
- `after_reliability_skip(ctx, phase, idx, candidate, score)`
- `after_candidate_probed(ctx, phase, idx, candidate, score, result)`
- `on_both_infeasible(ctx, phase, idx, candidate)`
- `after_phase_filter(ctx, label, before, after)`
- `after_branching_selection(ctx, candidate, score)`
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

function after_reliability_skip(
    ::BranchingContext, phase, idx::Int, candidate,
    score::Float64
)
    return
end

function after_candidate_probed(
    ::BranchingContext, phase, idx::Int, candidate,
    score::Float64, result
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
