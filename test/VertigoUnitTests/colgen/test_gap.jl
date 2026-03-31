# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_gap_decomposition_builder()
    @testset "[gap] decomposition builder" begin
        inst = random_gap_instance(2, 4; seed=42)
        ctx = build_gap_context(inst)

        @test length(collect(subproblem_ids(ctx.decomp))) == 2

        sp1 = PricingSubproblemId(1)
        @test length(subproblem_variables(ctx.decomp, sp1)) == 4
        lb1, ub1 = convexity_bounds(ctx.decomp, sp1)
        @test lb1 ≈ 0.0
        @test ub1 ≈ 1.0
        @test subproblem_fixed_cost(ctx.decomp, sp1) ≈ 0.0

        sp2 = PricingSubproblemId(2)
        @test length(subproblem_variables(ctx.decomp, sp2)) == 4
        lb2, ub2 = convexity_bounds(ctx.decomp, sp2)
        @test lb2 ≈ 0.0
        @test ub2 ≈ 1.0
        @test subproblem_fixed_cost(ctx.decomp, sp2) ≈ 0.0
        @test length(coupling_constraints(ctx.decomp)) == 4
        @test is_minimization(ctx.decomp)
    end
end

function test_gap_column_pool_populated()
    @testset "[gap] column pool is populated after CG" begin
        inst = random_gap_instance(2, 5; seed=42)
        ctx = build_gap_context(inst)

        run_column_generation(ctx)

        # Pool must have columns — at least one per machine
        @test length(ctx.pool.by_column_var) >= 2
    end
end

function test_gap_lp_dual_bound_matches_primal()
    @testset "[gap] LP dual bound approximately equals primal at convergence" begin
        inst = random_gap_instance(2, 7; seed=42)
        ctx = build_gap_context(inst)

        output = run_column_generation(ctx)

        @test output.status == Vertigo.ColGen.optimal
        @test !isnothing(output.incumbent_dual_bound)
        @test !isnothing(output.master_lp_obj)

        gap = abs(output.master_lp_obj - output.incumbent_dual_bound)
        @test gap <= 1.0  # within 1 unit (tight for LP relaxation)
    end
end
