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
                   prev_gap, gap) -> Bool

Single pure stopping criterion for the cut-column-generation loop.
"""
function stop_cutcolgen(
    ctx::CutColGenContext,
    round::Int,
    nb_cuts::Int,
    cg_status::ColGen.ColGenStatus,
    prev_gap::Float64,
    gap::Float64
)::Bool
    round >= ctx.max_rounds && return true
    nb_cuts == 0 && return true
    cg_status != ColGen.optimal && return true
    prev_gap <= 0.0 && return true
    improvement = (prev_gap - gap) / abs(prev_gap)
    improvement < ctx.min_gap_improvement && return true
    return false
end

"""
    _colgen_gap(output) -> Float64

Relative gap between dual bound and master LP objective.
"""
function _colgen_gap(output::ColGen.ColGenOutput)::Float64
    db = output.incumbent_dual_bound
    pb = output.master_lp_obj
    (isnothing(db) || isnothing(pb)) && return Inf
    iszero(pb) && return iszero(db) ? 0.0 : Inf
    return abs(pb - db) / abs(pb)
end
