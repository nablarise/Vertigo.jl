# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

module BranchCutPrice

using MathOptInterface
const MOI = MathOptInterface

using Printf
using ..Reformulation
using ..ColGen
using ..TreeSearch
using ..MathOptState
using ..Branching
using ..Branching: bp_decomp, bp_pool, bp_branching_constraints,
    bp_ip_incumbent, bp_ip_primal_bound,
    bp_set_ip_primal_bound!, bp_set_ip_incumbent!,
    bp_master_model, bp_robust_cuts,
    most_fractional_original_variable

include("bp_output.jl")
include("branching.jl")
include("cut_col_gen.jl")
include("space.jl")
include("cut_separation.jl")
include("evaluator.jl")
include("rmp_heuristic.jl")
include("dot_logger.jl")

export BPSpace, BPEvaluator, BPOutput, run_branch_and_price
export BPDotLoggerContext

end # module BranchCutPrice
