# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    AbstractBranchingRule

Determines candidate ordering for branching variable selection.
"""
abstract type AbstractBranchingRule end

"""
    MostFractionalRule <: AbstractBranchingRule

Select candidates closest to 0.5 fractionality first.
"""
struct MostFractionalRule <: AbstractBranchingRule end

"""
    LeastFractionalRule <: AbstractBranchingRule

Select candidates closest to integrality first.
"""
struct LeastFractionalRule <: AbstractBranchingRule end

"""
    select_candidates(rule, candidates, max_candidates)

Return the top `max_candidates` from `candidates` ordered by `rule`.
Input `candidates` is assumed sorted by fractionality descending.
"""
function select_candidates(
    ::MostFractionalRule,
    candidates::Vector{<:BranchingCandidate},
    max_candidates::Int
)
    return candidates[1:min(length(candidates), max_candidates)]
end

function select_candidates(
    ::LeastFractionalRule,
    candidates::Vector{<:BranchingCandidate},
    max_candidates::Int
)
    sorted = sort(candidates; by=c -> c.fractionality)
    return sorted[1:min(length(sorted), max_candidates)]
end
