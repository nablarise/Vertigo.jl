# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

function test_branch_and_price()
    @testset "[branch_and_price] small GAP finds integer solution" begin
        inst = random_gap_instance(2, 4; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp, BranchCutPriceConfig(node_limit = 100)
        )
        output = run_branch_and_price(bcp_ws)
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
        @test output.nodes_explored >= 1
    end

    @testset "[branch_and_price] respects node limit" begin
        inst = random_gap_instance(3, 8; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp, BranchCutPriceConfig(node_limit = 5)
        )
        output = run_branch_and_price(bcp_ws)
        @test output.nodes_explored <= 5
    end

    @testset "[branch_and_price] dual bound is valid" begin
        inst = random_gap_instance(2, 5; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp, BranchCutPriceConfig(node_limit = 50)
        )
        output = run_branch_and_price(bcp_ws)
        if !isnothing(output.incumbent)
            @test output.best_dual_bound <=
                  output.incumbent.obj_value + 1e-6
        end
    end

    @testset "[branch_and_price] logger runs without error" begin
        inst = random_gap_instance(2, 4; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp,
            BranchCutPriceConfig(node_limit = 100, log_level = 1)
        )
        output = redirect_stdout(devnull) do
            run_branch_and_price(bcp_ws)
        end
        @test output.status in (:optimal, :node_limit)
        @test output.nodes_explored >= 1
    end

    @testset "[branch_and_price] logger result matches non-logger" begin
        inst = random_gap_instance(2, 4; seed=42)
        decomp1 = build_gap_decomp(inst)
        out1 = run_branch_and_price(
            BranchCutPriceWorkspace(
                decomp1, BranchCutPriceConfig(node_limit = 50)
            )
        )

        decomp2 = build_gap_decomp(inst)
        out2 = redirect_stdout(devnull) do
            run_branch_and_price(BranchCutPriceWorkspace(
                decomp2,
                BranchCutPriceConfig(node_limit = 50, log_level = 1)
            ))
        end

        @test out1.status == out2.status
        @test out1.nodes_explored == out2.nodes_explored
    end
end
