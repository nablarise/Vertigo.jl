# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module BranchCutPrice

using MathOptInterface
const MOI = MathOptInterface

using Printf
using ..Reformulation
using ..ColGen
using ..TreeSearch
using ..MathOptState

include("interface.jl")
include("bp_output.jl")
include("branching.jl")
include("branching_candidates.jl")
include("branching_rules.jl")
include("branching_strategy.jl")
include("cut_col_gen.jl")
include("space.jl")
include("strong_branching.jl")
include("pseudocosts.jl")
include("reliability_branching.jl")
include("cut_separation.jl")
include("evaluator.jl")
include("rmp_heuristic.jl")
include("dot_logger.jl")

export BPSpace, BPEvaluator, BPOutput, run_branch_and_price
export BPDotLoggerContext
export AbstractBranchingStrategy, MostFractionalBranching, StrongBranching, ReliabilityBranching
export AbstractBranchingRule, MostFractionalRule, LeastFractionalRule
export BranchingStatus, BranchingResult
export branching_ok, all_integral, node_infeasible
export PseudocostRecord, PseudocostTracker
export update_pseudocosts!, estimate_score, is_reliable, global_average_pseudocost

end # module BranchCutPrice
