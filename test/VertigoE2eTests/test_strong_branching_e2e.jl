# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_strong_branching_e2e()
    filepath = get_gap_instance_path('A', 10, 100)
    inst = parse_gap_file(filepath)

    @testset "[bp][A 10×100] StrongBranching" begin
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx;
            node_limit=20,
            branching_strategy=StrongBranching(
                max_candidates=3, max_cg_iterations=5
            ),
            log_level=2
        )
        @test output.status in (:optimal, :node_limit)
        @test output.nodes_explored >= 2
    end

    @testset "[bp][A 10×100] ReliabilityBranching" begin
        rb = ReliabilityBranching(
            max_candidates=10, max_cg_iterations=5,
            reliability_threshold=4
        )
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx;
            node_limit=20,
            branching_strategy=rb,
            log_level=2
        )
        @test output.status in (:optimal, :node_limit)
        @test output.nodes_explored >= 2
        @test !isempty(rb.pseudocosts.records)
    end
end
