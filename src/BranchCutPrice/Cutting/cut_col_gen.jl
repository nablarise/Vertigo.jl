# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    CutSeparationConfig

Parameters of the cut-and-column-generation loop at each node.

# Fields
- `separator`: Robust cut separator, or `nothing` for no cut separation
  (default: `nothing`).
- `max_cut_rounds::Int`: Maximum cut separation rounds per node
  (default: 0).
- `min_gap_improvement::Float64`: Minimum relative improvement of the
  master LP value to continue cut rounds (default: 0.01).
"""
struct CutSeparationConfig
    separator::Union{Nothing,AbstractCutSeparator}
    max_cut_rounds::Int
    min_gap_improvement::Float64
    function CutSeparationConfig(;
        separator::Union{Nothing,AbstractCutSeparator}=nothing,
        max_cut_rounds::Int=0,
        min_gap_improvement::Float64=0.01,
    )
        return new(separator, max_cut_rounds, min_gap_improvement)
    end
end

"""
    stop_cutcolgen(ctx, round, nb_cuts, cg_status,
                   prev_lp, lp) -> Bool

Stopping criterion for the cut-column-generation loop. Tracks
master LP objective between rounds and stops when improvement
falls below `min_gap_improvement`.
"""
function stop_cutcolgen(
    ctx::CutSeparationConfig,
    round::Int,
    nb_cuts::Int,
    cg_status::ColGen.ColGenStatus,
    prev_lp::Float64,
    lp::Float64,
)::Bool
    if round >= ctx.max_cut_rounds
        return true
    end
    if nb_cuts == 0
        return true
    end
    if cg_status != ColGen.optimal
        return true
    end
    if isinf(prev_lp)
        return false
    end
    if iszero(prev_lp)
        return iszero(lp)
    end
    improvement = abs(lp - prev_lp) / abs(prev_lp)
    if improvement < ctx.min_gap_improvement
        return true
    end
    return false
end

"""
    _master_lp_obj(output) -> Float64

Extract master LP objective from CG output, or `Inf` if
unavailable.
"""
function _master_lp_obj(output::ColGen.ColGenOutput)::Float64
    if isnothing(output.master_lp_obj)
        return Inf
    end
    return output.master_lp_obj
end
