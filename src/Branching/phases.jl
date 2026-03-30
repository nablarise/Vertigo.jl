# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    AbstractBranchingPhase

Abstract supertype for phases in a multi-phase strong branching kernel.

Each phase evaluates a set of branching candidates and filters them down
to a smaller set for the next (more expensive) phase.
"""
abstract type AbstractBranchingPhase end

"""
    phase_label(phase::AbstractBranchingPhase) -> String

Return a short string label identifying the phase, used for logging.
"""
function phase_label(phase::AbstractBranchingPhase)
    error("phase_label not implemented for $(typeof(phase))")
end

"""
    LPProbePhase(; keep_fraction=0.25, lookahead=0)

A branching phase that scores candidates by solving LP relaxations.

No reliability skip — LP probes are inexpensive enough to always run.

## Fields
- `keep_fraction`: fraction of candidates to forward to the next phase
- `lookahead`: stop probing after this many candidates without
  improvement (0 = no early stopping)
"""
struct LPProbePhase <: AbstractBranchingPhase
    keep_fraction::Float64
    lookahead::Int

    function LPProbePhase(; keep_fraction::Float64=0.25, lookahead::Int=0)
        return new(keep_fraction, lookahead)
    end
end

"""
    phase_label(::LPProbePhase) -> String

Return `"LP"`.
"""
phase_label(::LPProbePhase) = "LP"

"""
    CGProbePhase(; max_cg_iterations=10, keep_fraction=1.0, lookahead=8)

A branching phase that scores candidates by running column generation probes.

## Fields
- `max_cg_iterations`: maximum CG iterations allowed per probe
- `keep_fraction`: fraction of candidates to forward to the next phase
  (1.0 = keep all, i.e. this is typically the final phase)
- `lookahead`: stop probing after this many candidates without
  improvement (0 = no early stopping)
"""
struct CGProbePhase <: AbstractBranchingPhase
    max_cg_iterations::Int
    keep_fraction::Float64
    lookahead::Int

    function CGProbePhase(;
        max_cg_iterations::Int=10,
        keep_fraction::Float64=1.0,
        lookahead::Int=8,
    )
        return new(max_cg_iterations, keep_fraction, lookahead)
    end
end

"""
    phase_label(::CGProbePhase) -> String

Return `"CG"`.
"""
phase_label(::CGProbePhase) = "CG"
