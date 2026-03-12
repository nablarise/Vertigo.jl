# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    CutColGenContext

Parameters for the cut-and-column-generation loop at each node.
`max_rounds = 0` means column generation only (no cut separation).
"""
struct CutColGenContext
    max_rounds::Int
    min_gap_improvement::Float64
end

"""
    stop_cutcolgen(ctx, round, nb_cuts, cg_status,
                   prev_lp, lp) -> Bool

Stopping criterion for the cut-column-generation loop. Tracks
master LP objective between rounds and stops when improvement
falls below `min_gap_improvement`.
"""
function stop_cutcolgen(
    ctx::CutColGenContext,
    round::Int,
    nb_cuts::Int,
    cg_status::ColGen.ColGenStatus,
    prev_lp::Float64,
    lp::Float64
)::Bool
    round >= ctx.max_rounds && return true
    nb_cuts == 0 && return true
    cg_status != ColGen.optimal && return true
    isinf(prev_lp) && return false
    iszero(prev_lp) && return iszero(lp)
    improvement = abs(lp - prev_lp) / abs(prev_lp)
    improvement < ctx.min_gap_improvement && return true
    return false
end

"""
    _master_lp_obj(output) -> Float64

Extract master LP objective from CG output, or `Inf` if
unavailable.
"""
function _master_lp_obj(
    output::ColGen.ColGenOutput
)::Float64
    isnothing(output.master_lp_obj) && return Inf
    return output.master_lp_obj
end
