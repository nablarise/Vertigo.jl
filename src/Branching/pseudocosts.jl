# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ────────────────────────────────────────────────────────────────────────────────────────
# PSEUDOCOST TRACKING
# ────────────────────────────────────────────────────────────────────────────────────────

mutable struct PseudocostRecord
    sum_down::Float64
    count_down::Int
    sum_up::Float64
    count_up::Int
end

PseudocostRecord() = PseudocostRecord(0.0, 0, 0.0, 0)

struct PseudocostTracker{X}
    records::Dict{X,PseudocostRecord}
    reliability_threshold::Int
end

function PseudocostTracker{X}(;
    reliability_threshold::Int = 8
) where {X}
    return PseudocostTracker{X}(
        Dict{X,PseudocostRecord}(), reliability_threshold
    )
end

"""
    update_pseudocosts!(tracker, candidate, result)

Update pseudocost records from a probe result. Skips
infeasible sides and sides with no dual bound.
"""
function update_pseudocosts!(
    tracker::PseudocostTracker,
    candidate::BranchingCandidate,
    result::SBCandidateResult
)
    rec = get!(
        tracker.records, candidate.orig_var, PseudocostRecord()
    )
    frac = candidate.value - candidate.floor_val

    if !result.left.is_infeasible &&
       !isnothing(result.left.dual_bound)
        delta = max(0.0, result.left.dual_bound - result.parent_lp_obj)
        rec.sum_down += delta / frac
        rec.count_down += 1
    end

    up_frac = 1.0 - frac
    if !result.right.is_infeasible &&
       !isnothing(result.right.dual_bound)
        delta = max(0.0, result.right.dual_bound - result.parent_lp_obj)
        rec.sum_up += delta / up_frac
        rec.count_up += 1
    end
    return
end

"""
    global_average_pseudocost(tracker) -> (avg_down, avg_up)

Compute the global average unit pseudocost across all variables
with observations. Returns `(1.0, 1.0)` if no observations exist
(Achterberg §2.2).
"""
function global_average_pseudocost(tracker::PseudocostTracker)
    total_down = 0.0
    n_down = 0
    total_up = 0.0
    n_up = 0
    for (_, rec) in tracker.records
        if rec.count_down > 0
            total_down += rec.sum_down / rec.count_down
            n_down += 1
        end
        if rec.count_up > 0
            total_up += rec.sum_up / rec.count_up
            n_up += 1
        end
    end
    avg_down = n_down > 0 ? total_down / n_down : 1.0
    avg_up = n_up > 0 ? total_up / n_up : 1.0
    return avg_down, avg_up
end

"""
    estimate_score(tracker, candidate; mu=1/6) -> Float64

Estimate branching score from pseudocosts. Uses global average
as fallback for variables with no observations on one or both
sides (Achterberg et al., 2005, §2.2).
"""
function estimate_score(
    tracker::PseudocostTracker,
    candidate::BranchingCandidate;
    mu::Float64 = 1.0 / 6.0
)
    frac = candidate.value - candidate.floor_val
    avg_down, avg_up = global_average_pseudocost(tracker)

    if haskey(tracker.records, candidate.orig_var)
        rec = tracker.records[candidate.orig_var]
        mean_down = rec.count_down > 0 ?
            rec.sum_down / rec.count_down : avg_down
        mean_up = rec.count_up > 0 ?
            rec.sum_up / rec.count_up : avg_up
    else
        mean_down = avg_down
        mean_up = avg_up
    end

    score_down = mean_down * frac
    score_up = mean_up * (1.0 - frac)

    return (1 - mu) * min(score_down, score_up) +
           mu * max(score_down, score_up)
end

"""
    is_reliable(tracker, candidate) -> Bool

True when `min(count_down, count_up) >= reliability_threshold`.
"""
function is_reliable(
    tracker::PseudocostTracker,
    candidate::BranchingCandidate
)
    !haskey(tracker.records, candidate.orig_var) && return false
    rec = tracker.records[candidate.orig_var]
    return min(rec.count_down, rec.count_up) >=
           tracker.reliability_threshold
end
