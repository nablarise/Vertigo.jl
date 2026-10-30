# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ── Time limit e2e tests on GAP class A ──────────────────────────────────

function test_time_limit_e2e()
    inst = parse_gap_file(get_gap_instance_path('A', 20, 100))

    @testset "[time limit] column generation stops with status time_limit" begin
        decomp = build_gap_decomp(inst)
        output = run_col_gen(
            decomp, ColGenConfig(time_limit=0.0, silent=true)
        )
        @test output.status == time_limit
    end

    @testset "[time limit] column generation without limit converges" begin
        decomp = build_gap_decomp(inst)
        output = run_col_gen(decomp, ColGenConfig(silent=true))
        @test output.status == optimal
    end

    @testset "[time limit] branch-and-price with an expired limit" begin
        bcp_ws = BranchCutPriceWorkspace(
            build_gap_decomp(inst),
            BranchCutPriceConfig(time_limit=0.0)
        )
        output = run_branch_and_price(bcp_ws)
        @test output.status == :time_limit
        @test isnothing(output.incumbent)
    end

    @testset "[time limit] branch-and-price stops near the limit" begin
        limit = 3.0
        bcp_ws = BranchCutPriceWorkspace(
            build_gap_decomp(inst),
            BranchCutPriceConfig(
                colgen=ColGenConfig(smoothing_alpha=0.5, silent=true),
                node_limit=1_000_000, time_limit=limit
            )
        )
        elapsed = @elapsed output = run_branch_and_price(bcp_ws)
        @test output.status in (:time_limit, :optimal)
        @test elapsed < limit + 10.0
        @test output.best_dual_bound <= 1158 + 1e-4
    end
end
