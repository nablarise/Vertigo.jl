# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_branch_and_price()
    @testset "[branch_and_price] small GAP finds integer solution" begin
        inst = random_gap_instance(2, 4; seed=42)
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx; node_limit = 100
        )
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
        @test output.nodes_explored >= 1
    end

    @testset "[branch_and_price] respects node limit" begin
        inst = random_gap_instance(3, 8; seed=42)
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx; node_limit = 5
        )
        @test output.nodes_explored <= 5
    end

    @testset "[branch_and_price] dual bound is valid" begin
        inst = random_gap_instance(2, 5; seed=42)
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx; node_limit = 50
        )
        if !isnothing(output.incumbent)
            @test output.best_dual_bound <=
                  output.incumbent.obj_value + 1e-6
        end
    end
end
