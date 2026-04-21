# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.ColGen: ColGenLoggerWorkspace, max_cg_iterations,
    set_max_cg_iterations!, stop_colgen_phase,
    ColGenIterationOutput, Phase0, Phase1, Phase2

function test_max_cg_iterations()
    @testset "[max_cg_iterations] stop_colgen_phase respects limit" begin
        inst = random_gap_instance(2, 5; seed=42)
        ws = build_gap_context(inst; max_cg_iterations=5)

        # Fake iteration output: columns added, no gap closed
        iter_output = ColGenIterationOutput(
            100.0,   # master_lp_obj
            nothing, # dual_bound
            1,       # nb_columns_added (not zero)
            nothing, # master_lp_dual_sol
            nothing, # master_ip_primal_sol
            false    # subproblem_infeasible
        )

        # Phase0: iteration 4 (≤ 5) → don't stop
        @test !stop_colgen_phase(ws, Phase0(), iter_output, nothing, nothing, 4)
        # Phase0: iteration 6 (> 5) → stop
        @test stop_colgen_phase(ws, Phase0(), iter_output, nothing, nothing, 6)
        # Phase2: same behavior
        @test !stop_colgen_phase(ws, Phase2(), iter_output, nothing, nothing, 5)
        @test stop_colgen_phase(ws, Phase2(), iter_output, nothing, nothing, 6)
    end

    @testset "[max_cg_iterations] Phase1 ignores limit" begin
        inst = random_gap_instance(2, 5; seed=42)
        ws = build_gap_context(inst; max_cg_iterations=5)

        # Phase1 iter output: columns still being added
        iter_output = ColGenIterationOutput(
            100.0, nothing, 1, nothing, nothing, false
        )

        # Phase1 does not stop even beyond limit
        @test !stop_colgen_phase(ws, Phase1(), iter_output, nothing, nothing, 100)
    end

    @testset "[max_cg_iterations] ColGenLoggerWorkspace forwarding" begin
        inst = random_gap_instance(2, 5; seed=42)
        ws = build_gap_context(inst; max_cg_iterations=42)
        lws = ColGenLoggerWorkspace(ws; log_level=0)
        @test max_cg_iterations(lws) == 42
        set_max_cg_iterations!(lws, 7)
        @test max_cg_iterations(lws) == 7
        # Verify it changed the inner context too
        @test max_cg_iterations(ws) == 7
    end
end
