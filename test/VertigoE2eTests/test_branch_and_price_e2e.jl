# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Branch-and-price e2e tests on GAP class A ────────────────────────────

# (class, agents, jobs, optimal_value)
const GAP_A_BP_INSTANCES = [
    #('A', 5,  100, 1698),
    #('A', 5,  200, 3235),
    ('A', 10, 100, 1360),
    #('A', 10, 200, 2623),
    ('A', 20, 100, 1158),
    #('A', 20, 200, 2339)
]

function test_bp_gap_a_instances()
    for (class, agents, jobs, optimal) in GAP_A_BP_INSTANCES
        label = "$(uppercase(class)) $(agents)×$(jobs)"
        @testset "[bp][$(label)] optimal = $(optimal)" begin
            filepath = get_gap_instance_path(class, agents, jobs)
            inst = parse_gap_file(filepath)
            ctx = build_gap_context(inst; smoothing_alpha=0.5)
            dot_path = "$class-$agents-$jobs.dot"
            output = run_branch_and_price(
                ctx; node_limit = 5_000,
                dot_file = dot_path
            )

            @test output.status in (:optimal, :node_limit)

            # Dual bound must be valid (≤ optimal for min)
            @test output.best_dual_bound <= optimal + 1e-4

            if !isnothing(output.incumbent)
                # Incumbent is a valid upper bound
                @test output.incumbent.obj_value >=
                    optimal - 1e-4

                # Dual bound ≤ incumbent (valid relaxation)
                @test output.best_dual_bound <=
                    output.incumbent.obj_value + 1e-4
            end

            if output.status == :optimal
                @test !isnothing(output.incumbent)
                @test abs(output.incumbent.obj_value - optimal) <=
                    1e-4
            end

            # Dot file structural validity
            @test isfile(dot_path)
            dot_content = read(dot_path, String)
            @test startswith(dot_content, "digraph BCP_Tree {")
            @test endswith(strip(dot_content), "}")
            @test occursin(r"n\d+ \[label=", dot_content)
        end
    end
end
