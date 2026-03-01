# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Branch-and-price e2e tests ───────────────────────────────────────────────
# Each test mirrors its CG counterpart but runs through run_branch_and_price.

function test_bp_gap_converges()
    @testset "[bp] converges (2 machines, 7 tasks)" begin
        inst = random_gap_instance(2, 7)
        ctx = build_gap_context(inst)
        output = run_branch_and_price(ctx; node_limit = 200)
        @test output.status in (:optimal, :node_limit)
        if !isnothing(output.incumbent)
            @test output.best_dual_bound <=
                output.incumbent.obj_value + 1e-4
        end
    end
end

function test_bp_gap_benchmark_instances()
    for (class, agents, jobs, expected_bound) in BENCHMARK_INSTANCES
        label = "$(uppercase(class)) $(agents)×$(jobs)"
        @testset "[bp][$(label)] dual bound ≈ $(expected_bound)" begin
            filepath = get_gap_instance_path(class, agents, jobs)
            inst = parse_gap_file(filepath)
            ctx = build_gap_context(inst)
            output = run_branch_and_price(ctx; node_limit = 500)
            @test output.status in (:optimal, :node_limit)
            if !isnothing(output.incumbent)
                @test output.best_dual_bound <=
                    output.incumbent.obj_value + 1e-4
            end
        end
    end
end

function test_bp_gap_infeasible_master()
    @testset "[bp] infeasible master (3 machines, 30 jobs)" begin
        inst = gap_infeasible_master()
        ctx = build_gap_context(inst)
        output = run_branch_and_price(ctx; node_limit = 50)
        @test output.status == :infeasible
    end
end

function test_bp_gap_infeasible_subproblem()
    @testset "[bp] infeasible subproblem (machine 1 capacity = -1)" begin
        inst = gap_infeasible_subproblem()
        ctx = build_gap_context(inst)
        output = run_branch_and_price(ctx; node_limit = 50)
        @test output.status == :infeasible
    end
end

function test_bp_gap_maximization()
    @testset "[bp] maximization (3 machines, 30 jobs)" begin
        inst = gap_maximization()
        ctx = build_gap_context_max(inst)
        output = run_branch_and_price(ctx; node_limit = 200)
        @test output.status in (:optimal, :node_limit, :infeasible)
        if !isnothing(output.incumbent)
            @test output.best_dual_bound >=
                output.incumbent.obj_value - 1e-4
        end
    end
end

function test_bp_gap_two_identical_machines()
    @testset "[bp] two identical machines (symmetric degeneracy)" begin
        inst = gap_two_identical_machines()
        ctx = build_gap_context(inst)
        output = run_branch_and_price(ctx; node_limit = 200)
        @test output.status in (:optimal, :node_limit, :infeasible)
    end
end

function test_bp_gap_three_identical_machines()
    @testset "[bp] three identical machines (maximum symmetry)" begin
        inst = gap_three_identical_machines()
        ctx = build_gap_context(inst)
        output = run_branch_and_price(ctx; node_limit = 200)
        @test output.status in (:optimal, :node_limit, :infeasible)
    end
end
