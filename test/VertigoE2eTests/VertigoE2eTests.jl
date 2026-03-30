# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module VertigoE2eTests

using Test
using Random
using JuMP
using HiGHS
using MathOptInterface
using ZipFile
const MOI = MathOptInterface

using Vertigo
using Vertigo.Branching: MultiPhaseStrongBranching, CGProbePhase

include("gap_decomposition.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# Column generation e2e tests
# ────────────────────────────────────────────────────────────────────────────────────────

include("test_column_generation_e2e.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# Branch-and-price e2e tests
# ────────────────────────────────────────────────────────────────────────────────────────

include("test_branch_and_price_e2e.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# DML e2e tests
# ────────────────────────────────────────────────────────────────────────────────────────

include("test_dml_e2e.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# Branch-cut-price with cuts e2e tests
# ────────────────────────────────────────────────────────────────────────────────────────

include("test_cut_col_gen_e2e.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# Strong branching e2e tests
# ────────────────────────────────────────────────────────────────────────────────────────

include("test_strong_branching_e2e.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function run()
    test_column_generation_e2e()
    test_bp_gap_a_instances()
    test_dml_e2e()
    test_cut_col_gen_e2e()
    test_strong_branching_e2e()
end

end # module VertigoE2eTests
