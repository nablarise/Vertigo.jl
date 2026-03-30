# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module Branching

using MathOptInterface
const MOI = MathOptInterface
using Printf

using ..Reformulation
using ..ColGen
using ..MathOptState

include("interface.jl")
include("branching_utils.jl")
include("branching_candidates.jl")
include("branching_rules.jl")
include("branching_strategy.jl")
include("phases.jl")
include("strong_branching.jl")
include("lp_probe.jl")
include("cg_probe.jl")
include("pseudocosts.jl")
include("kernel.jl")

# ── Public API ────────────────────────────────────────────────────────────
export AbstractBranchingPhase, LPProbePhase, CGProbePhase
export AbstractBranchingStrategy, MostFractionalBranching
export MultiPhaseStrongBranching
export AbstractBranchingRule, MostFractionalRule, LeastFractionalRule
export BranchingStatus, BranchingResult
export branching_ok, all_integral, node_infeasible
export BranchingDirection, branch_down, branch_up
export select_branching_variable, on_node_evaluated

end # module Branching
