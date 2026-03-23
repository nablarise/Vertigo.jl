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
include("strong_branching.jl")
include("pseudocosts.jl")
include("reliability_branching.jl")

export AbstractBranchingStrategy, MostFractionalBranching
export StrongBranching, ReliabilityBranching
export AbstractBranchingRule, MostFractionalRule, LeastFractionalRule
export BranchingStatus, BranchingResult
export branching_ok, all_integral, node_infeasible
export BranchingDirection, branch_down, branch_up
export BranchingCandidate, find_fractional_variables
export SBProbeResult, SBCandidateResult, sb_score
export PseudocostRecord, PseudocostTracker
export update_pseudocosts!, estimate_score, is_reliable
export global_average_pseudocost
export select_branching_variable, on_node_evaluated
export most_fractional_original_variable
export build_branching_terms, add_branching_constraint!
export remove_branching_constraint!, run_sb_probe
export bp_decomp, bp_pool, bp_branching_constraints
export bp_ip_incumbent, bp_ip_primal_bound
export bp_set_ip_primal_bound!, bp_set_ip_incumbent!
export bp_master_model, bp_robust_cuts

end # module Branching
