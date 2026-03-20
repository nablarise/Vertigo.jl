# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.BranchCutPrice: find_fractional_variables,
    BranchingCandidate

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
end
