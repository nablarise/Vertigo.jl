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
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function run()
    test_column_generation_e2e()
    println("---------")
    test_bp_gap_a_instances()
    println("---------")
    test_dml_e2e()
end

end # module VertigoE2eTests
