# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Branch-and-price e2e tests ───────────────────────────────────────────────

function test_bp_gap_benchmark_instances()
    for (class, agents, jobs, expected_bound) in BENCHMARK_INSTANCES
        label = "$(uppercase(class)) $(agents)×$(jobs)"
        @testset "[bp][$(label)] dual bound ≈ $(expected_bound)" begin
            filepath = get_gap_instance_path(class, agents, jobs)
            inst = parse_gap_file(filepath)
            ctx = build_gap_context(inst)
            output = run_branch_and_price(ctx; node_limit = 500, log = true)
            @test output.status in (:optimal, :node_limit)
            if !isnothing(output.incumbent)
                @test output.best_dual_bound <=
                    output.incumbent.obj_value + 1e-4
            end
        end
    end
end
