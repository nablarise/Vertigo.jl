# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_strong_branching_e2e()
    filepath = get_gap_instance_path('A', 10, 100)
    inst = parse_gap_file(filepath)

    @testset "[bp][A 10×100] MultiPhaseStrongBranching" begin
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx;
            node_limit=20,
            branching_strategy=MultiPhaseStrongBranching(
                max_candidates=3,
                phases=[CGProbePhase(
                    max_cg_iterations=5, lookahead=0
                )]
            ),
            log_level=2
        )
        @test output.status in (:optimal, :node_limit)
        @test output.nodes_explored >= 2
    end

    @testset "[bp][A 10×100] MultiPhaseStrongBranching with reliability" begin
        rb = MultiPhaseStrongBranching(
            max_candidates=10,
            phases=[CGProbePhase(
                max_cg_iterations=5, lookahead=8
            )],
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
        # Broken: run_branch_and_price reconstructs the strategy,
        # so rb.pseudocosts is never updated. See #37.
        @test_broken !isempty(rb.pseudocosts.records)
    end
end
