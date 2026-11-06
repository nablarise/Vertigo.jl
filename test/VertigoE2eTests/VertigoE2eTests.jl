# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

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
# Time limit e2e tests
# ────────────────────────────────────────────────────────────────────────────────────────

include("test_time_limit_e2e.jl")
include("test_artificial_vars_e2e.jl")
include("test_dual_bound_e2e.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

# See VertigoUnitTests.run: a failing test must not hide the ones after it.
function run()
    @testset "VertigoE2eTests" begin
        for test in (
            test_column_generation_e2e,
            test_bp_gap_a_instances,
            test_bp_ip_incumbent_before_convergence,
            test_bp_branching_constraint_coefficients,
            test_dml_e2e,
            test_cut_col_gen_e2e,
            test_time_limit_e2e,
            test_artificial_vars_e2e,
            test_dual_bound_e2e,
        )
            @testset "$(nameof(test))" begin
                test()
            end
        end
    end
    return nothing
end

end # module VertigoE2eTests
