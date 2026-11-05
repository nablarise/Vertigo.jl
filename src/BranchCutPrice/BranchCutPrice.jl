# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

module BranchCutPrice

using MathOptInterface
const MOI = MathOptInterface

using Printf
using ..Decomposition
using ..ColGen
using ..TreeSearch
using ..MathOptState

include("Branching/Branching.jl")
using .Branching
using .Branching:
    bp_decomp,
    bp_pool,
    bp_branching_constraints,
    bp_ip_incumbent,
    bp_ip_primal_bound,
    bp_set_ip_primal_bound!,
    bp_set_ip_incumbent!,
    bp_master_model,
    bp_robust_cuts,
    most_fractional_original_variable

include("Cutting/Cutting.jl")
using .Cutting
using .Cutting: _separate_and_add_cuts!, _master_lp_obj

include("bp_output.jl")
include("branching.jl")
include("config.jl")
include("space.jl")
include("evaluator.jl")
include("rmp_heuristic.jl")
include("logger.jl")

export BPEvaluator, BPOutput
export BranchCutPriceConfig, BranchCutPriceWorkspace, CutSeparationConfig
export AbstractCutSeparator, SeparatedCut, separate, CustomCutSeparator
export run_branch_and_price
export parent_lp_obj, branching_var, branching_direction, branching_frac
export BPDotLoggerContext

end # module BranchCutPrice
