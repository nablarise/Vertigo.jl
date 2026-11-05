# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

module Cutting

using MathOptInterface
const MOI = MathOptInterface

using ...Decomposition
using ...ColGen
using ..Branching: bp_decomp, bp_pool, bp_robust_cuts

include("interface.jl")
include("cut_col_gen.jl")
include("cut_separation.jl")

# ── Public API ────────────────────────────────────────────────────────────
export AbstractCutSeparator, SeparatedCut, separate, CustomCutSeparator
export CutSeparationConfig, stop_cutcolgen

end # module Cutting
