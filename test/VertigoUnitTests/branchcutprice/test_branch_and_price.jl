# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# A best-first strategy ordering the nodes by the BCP data of a `NodeView`.
struct DeepestFirstByView <: Vertigo.TreeSearch.AbstractBestFirstStrategy end
const VIEWED_NODES = Tuple{Int,Any,Any}[]
function Vertigo.TreeSearch.get_priority(::DeepestFirstByView, node::Vertigo.TreeSearch.NodeView)
    push!(
        VIEWED_NODES,
        (node.depth, Vertigo.branching_direction(node), Vertigo.parent_lp_obj(node)),
    )
    return -Float64(node.depth)
end

function test_branch_and_price()
    @testset "[branch_and_price] a priority function reads the BCP data of a node view" begin
        empty!(VIEWED_NODES)
        inst = random_gap_instance(2, 4; seed=42)
        ws = BranchCutPriceWorkspace(
            build_gap_decomp(inst),
            BranchCutPriceConfig(node_limit=100, node_select_strategy=DeepestFirstByView()),
        )
        output = run_branch_and_price(ws)
        @test !isnothing(output.incumbent)
        @test VIEWED_NODES[1] == (0, nothing, nothing)   # the root has no branching data
        @test all(t -> t[1] == 0 || !isnothing(t[2]), VIEWED_NODES)   # children know their direction
    end

    @testset "[branch_and_price] the node selection strategy must be a search strategy" begin
        config = BranchCutPriceConfig(node_select_strategy=Vertigo.TreeSearch.BreadthFirstStrategy())
        @test config.node_select_strategy isa Vertigo.TreeSearch.AbstractSearchStrategy
        @test_throws TypeError BranchCutPriceConfig(node_select_strategy=:depth_first)
    end

    @testset "[branch_and_price] small GAP finds integer solution" begin
        inst = random_gap_instance(2, 4; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(decomp, BranchCutPriceConfig(node_limit=100))
        output = run_branch_and_price(bcp_ws)
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
        @test output.nodes_explored >= 1
        @test !isnothing(output.root_lp_value)
        @test output.root_lp_value <= output.incumbent.obj_value + 1e-6
    end

    @testset "[branch_and_price] respects node limit" begin
        inst = random_gap_instance(3, 8; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(decomp, BranchCutPriceConfig(node_limit=5))
        output = run_branch_and_price(bcp_ws)
        @test output.nodes_explored <= 5
    end

    @testset "[branch_and_price] dual bound is valid" begin
        inst = random_gap_instance(2, 5; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(decomp, BranchCutPriceConfig(node_limit=50))
        output = run_branch_and_price(bcp_ws)
        if !isnothing(output.incumbent)
            @test output.best_dual_bound <= output.incumbent.obj_value + 1e-6
        end
    end

    @testset "[branch_and_price] logger runs without error" begin
        inst = random_gap_instance(2, 4; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp, BranchCutPriceConfig(node_limit=100, log_level=1)
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
            BranchCutPriceWorkspace(decomp1, BranchCutPriceConfig(node_limit=50))
        )

        decomp2 = build_gap_decomp(inst)
        out2 = redirect_stdout(devnull) do
            run_branch_and_price(
                BranchCutPriceWorkspace(
                    decomp2, BranchCutPriceConfig(node_limit=50, log_level=1)
                ),
            )
        end

        @test out1.status == out2.status
        @test out1.nodes_explored == out2.nodes_explored
    end
end
