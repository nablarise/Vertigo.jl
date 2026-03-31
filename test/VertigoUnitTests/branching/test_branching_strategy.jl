# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.Branching: find_fractional_variables,
    BranchingCandidate, MostFractionalRule, LeastFractionalRule,
    select_candidates, MostFractionalBranching,
    select_branching_variable, bp_master_model,
    most_fractional_original_variable,
    BranchingResult, branching_ok, all_integral
using Vertigo.BranchCutPrice: BPSpace

function test_branching_strategy()
    @testset "[find_fractional_variables] all integral" begin
        inst = random_gap_instance(2, 5; seed=42)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)
        primal = Dict{MOI.VariableIndex,Float64}()
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        @test isempty(candidates)
    end

    @testset "[find_fractional_variables] detects fractional and sorts" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)
        primal = get_primal_solution(bp_master_model(ctx))
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        # GAP(2,5) always has fractional LP relaxation
        @test !isempty(candidates)
        c = first(candidates)
        @test c isa BranchingCandidate
        @test c.fractionality > 0.0
        @test c.floor_val == floor(c.value)
        @test c.ceil_val == ceil(c.value)
        # Sorted descending by fractionality
        @test length(candidates) >= 2
        @test candidates[1].fractionality >=
              candidates[2].fractionality
    end

    @testset "[branching_rules] MostFractionalRule ordering" begin
        # Manually create candidates with known fractionalities
        c1 = BranchingCandidate(1, 1.3, 1.0, 2.0, 0.3)
        c2 = BranchingCandidate(2, 2.5, 2.0, 3.0, 0.5)
        c3 = BranchingCandidate(3, 3.1, 3.0, 4.0, 0.1)
        candidates = [c2, c1, c3]  # already sorted desc
        result = select_candidates(
            MostFractionalRule(), candidates, 2
        )
        @test length(result) == 2
        @test result[1].fractionality ≈ 0.5
        @test result[2].fractionality ≈ 0.3
    end

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

    @testset "[branching_strategy] MostFractionalBranching delegates" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)
        primal = get_primal_solution(bp_master_model(ctx))

        # Compare strategy with direct call (same tol)
        orig_var, x_val = most_fractional_original_variable(
            ctx, primal; tol=1e-6
        )
        # Strategy needs a BPSpace — construct one
        space = BPSpace(ctx; node_limit=1)
        result = select_branching_variable(
            MostFractionalBranching(), space, nothing, primal
        )
        # GAP(2,5) with seed=10 always has fractional LP relaxation
        @test !isnothing(orig_var)
        @test result.status == branching_ok
        @test result.orig_var == orig_var
        @test result.value ≈ x_val
    end
end
