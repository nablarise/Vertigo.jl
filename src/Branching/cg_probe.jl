# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    probe_candidate(::CGProbePhase, space, candidate, parent_lp)

CG probe: run column generation with limited iterations in both
directions. Delegates to `run_sb_probe`.
"""
function probe_candidate(
    phase::CGProbePhase, space,
    candidate::BranchingCandidate, parent_lp::Float64
)
    return run_sb_probe(
        space, candidate, phase.max_cg_iterations, parent_lp
    )
end
