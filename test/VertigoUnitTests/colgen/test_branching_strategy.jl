# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.BranchCutPrice: find_fractional_variables,
    BranchingCandidate, MostFractionalRule, LeastFractionalRule, select_candidates

function test_branching_strategy()
    @testset "[find_fractional_variables] all integral" begin
        inst = random_gap_instance(2, 5)
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
        # Verify sorted descending by fractionality
        for i in 1:length(candidates)-1
            @test candidates[i].fractionality >=
                  candidates[i+1].fractionality
        end
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
        @test result[1].fractionality == 0.5
        @test result[2].fractionality == 0.3
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
        @test result[1].fractionality == 0.1
        @test result[2].fractionality == 0.3
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
end
