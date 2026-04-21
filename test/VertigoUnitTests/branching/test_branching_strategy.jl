# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.Branching: find_fractional_variables,
    BranchingCandidate, MostFractionalRule, LeastFractionalRule,
    select_candidates

# ────────────────────────────────────────────────────────────────
# Minimal context for find_fractional_variables
# ────────────────────────────────────────────────────────────────
#
# 1 subproblem with 3 sp variables (z1, z2, z3)
# mapped to original vars (1,1), (1,2), (1,3).
#
# Two columns in the pool:
#   Column A (col_var_a): z1=1, z2=1  — cost 3.0
#   Column B (col_var_b): z2=1, z3=1  — cost 5.0
#
# No solver, no master model, no CG needed.

function _build_branching_test_context()
    sp1 = PricingSubproblemId(1)
    z1 = MOI.VariableIndex(1)
    z2 = MOI.VariableIndex(2)
    z3 = MOI.VariableIndex(3)

    builder = DWReformulationBuilder{Tuple{Int,Int}}(
        minimize=true
    )
    add_subproblem!(builder, sp1, 0.0, 0.0, 1.0)
    add_sp_variable!(builder, sp1, z1, 3.0)
    add_sp_variable!(builder, sp1, z2, 4.0)
    add_sp_variable!(builder, sp1, z3, 5.0)
    add_mapping!(builder, (1, 1), sp1, z1)
    add_mapping!(builder, (1, 2), sp1, z2)
    add_mapping!(builder, (1, 3), sp1, z3)
    decomp = build(builder)

    pool = ColumnPool()
    col_var_a = MOI.VariableIndex(101)
    col_var_b = MOI.VariableIndex(102)

    sol_a = Vertigo.Reformulation._SpSolution(
        sp1, 3.0, [(z1, 1.0), (z2, 1.0)]
    )
    sol_b = Vertigo.Reformulation._SpSolution(
        sp1, 5.0, [(z2, 1.0), (z3, 1.0)]
    )
    record_column!(pool, col_var_a, sp1, sol_a, 3.0)
    record_column!(pool, col_var_b, sp1, sol_b, 5.0)

    config = ColGenConfig()
    ctx = ColGenWorkspace(decomp, pool,
        Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        config
    )

    return ctx, col_var_a, col_var_b
end

# ────────────────────────────────────────────────────────────────
# Tests — find_fractional_variables
# ────────────────────────────────────────────────────────────────

function test_find_fractional_all_integral()
    @testset "[find_fractional_variables] empty primal → no candidates" begin
        ctx, _, _ = _build_branching_test_context()
        primal = Dict{MOI.VariableIndex,Float64}()
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        @test isempty(candidates)
    end
end

function test_find_fractional_with_known_values()
    @testset "[find_fractional_variables] known λ → known fractional x" begin
        ctx, col_var_a, col_var_b = _build_branching_test_context()

        # λ_a = 0.3, λ_b = 0.6
        primal = Dict{MOI.VariableIndex,Float64}(
            col_var_a => 0.3,
            col_var_b => 0.6,
        )

        # Hand-computed projection x[orig] = Σ z_val * λ:
        #   x(1,1) = 1.0 * 0.3           = 0.3  → frac = 0.3
        #   x(1,2) = 1.0 * 0.3 + 1.0 * 0.6 = 0.9  → frac = 0.1
        #   x(1,3) = 1.0 * 0.6           = 0.6  → frac = 0.4
        #
        # Sorted desc by fractionality:
        #   (1,3) frac=0.4, (1,1) frac=0.3, (1,2) frac=0.1

        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )

        @test length(candidates) == 3

        @test candidates[1].orig_var == (1, 3)
        @test candidates[1].value ≈ 0.6 atol=1e-10
        @test candidates[1].fractionality ≈ 0.4 atol=1e-10
        @test candidates[1].floor_val ≈ 0.0 atol=1e-10
        @test candidates[1].ceil_val ≈ 1.0 atol=1e-10

        @test candidates[2].orig_var == (1, 1)
        @test candidates[2].value ≈ 0.3 atol=1e-10
        @test candidates[2].fractionality ≈ 0.3 atol=1e-10

        @test candidates[3].orig_var == (1, 2)
        @test candidates[3].value ≈ 0.9 atol=1e-10
        @test candidates[3].fractionality ≈ 0.1 atol=1e-10
    end
end

function test_find_fractional_integral_projection()
    @testset "[find_fractional_variables] λ=1.0 → integral projection" begin
        ctx, col_var_a, _ = _build_branching_test_context()

        # Only column A active at λ=1.0:
        #   x(1,1) = 1.0, x(1,2) = 1.0 — both integral
        primal = Dict{MOI.VariableIndex,Float64}(
            col_var_a => 1.0,
        )

        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        @test isempty(candidates)
    end
end

# ────────────────────────────────────────────────────────────────
# Tests — select_candidates (pure, no context needed)
# ────────────────────────────────────────────────────────────────

function test_select_candidates_most_fractional()
    @testset "[branching_rules] MostFractionalRule ordering" begin
        c1 = BranchingCandidate(1, 1.3, 1.0, 2.0, 0.3)
        c2 = BranchingCandidate(2, 2.5, 2.0, 3.0, 0.5)
        c3 = BranchingCandidate(3, 3.1, 3.0, 4.0, 0.1)
        candidates = [c2, c1, c3]
        result = select_candidates(
            MostFractionalRule(), candidates, 2
        )
        @test length(result) == 2
        @test result[1].fractionality ≈ 0.5
        @test result[2].fractionality ≈ 0.3
    end
end

function test_select_candidates_least_fractional()
    @testset "[branching_rules] LeastFractionalRule ordering" begin
        c1 = BranchingCandidate(1, 1.3, 1.0, 2.0, 0.3)
        c2 = BranchingCandidate(2, 2.5, 2.0, 3.0, 0.5)
        c3 = BranchingCandidate(3, 3.1, 3.0, 4.0, 0.1)
        candidates = [c2, c1, c3]
        result = select_candidates(
            LeastFractionalRule(), candidates, 2
        )
        @test length(result) == 2
        @test result[1].fractionality ≈ 0.1
        @test result[2].fractionality ≈ 0.3
    end
end

function test_select_candidates_truncation()
    @testset "[branching_rules] max_candidates truncation" begin
        cs = [
            BranchingCandidate(
                i, Float64(i) + 0.5, Float64(i),
                Float64(i) + 1.0, 0.5
            ) for i in 1:10
        ]
        result = select_candidates(MostFractionalRule(), cs, 3)
        @test length(result) == 3
    end
end

# ────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────

function test_branching_strategy()
    test_find_fractional_all_integral()
    test_find_fractional_with_known_values()
    test_find_fractional_integral_projection()
    test_select_candidates_most_fractional()
    test_select_candidates_least_fractional()
    test_select_candidates_truncation()
end
