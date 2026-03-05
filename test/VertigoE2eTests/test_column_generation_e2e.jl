# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# GAP instance constructors for CG tests
# ────────────────────────────────────────────────────────────────────────────────────────

function gap_infeasible_master()
    # 3 machines, 30 jobs, capacity 45, weight 5 per task.
    # Each machine covers at most 45/5 = 9 tasks → max 27 total < 30 required.
    n_machines = 3
    n_tasks = 30
    cost = fill(1.0, n_machines, n_tasks)
    weight = fill(5.0, n_machines, n_tasks)
    capacity = fill(45.0, n_machines)
    return GAPInstance(n_machines, n_tasks, cost, weight, capacity)
end

function gap_infeasible_subproblem()
    # Machine 1 has capacity = -1 → its subproblem constraint
    # Σ weight * z <= -1 is infeasible for z >= 0.
    n_machines = 3
    n_tasks = 30
    cost = fill(1.0, n_machines, n_tasks)
    weight = fill(5.0, n_machines, n_tasks)
    capacity = [-1.0, 45.0, 45.0]
    return GAPInstance(n_machines, n_tasks, cost, weight, capacity)
end

function gap_maximization()
    # Maximization GAP: negate gap_small_feasible2 costs.
    inst = gap_small_feasible2()
    cost = -inst.cost
    return GAPInstance(
        inst.n_machines, inst.n_tasks, cost, inst.weight, inst.capacity
    )
end

# Root dual bound is 32
# Optimal value is 32
function gap_two_identical_machines()
    # 3 machines, 7 tasks — machines 1 and 2 are identical
    cost = [5.0  8.0  3.0  12.0  7.0  4.0  9.0;
            5.0  8.0  3.0  12.0  7.0  4.0  9.0;
            8.0  6.0  10.0  4.0  11.0  7.0  3.0]
    weight = [2.0  3.0  1.0  4.0  2.0  1.0  3.0;
              2.0  3.0  1.0  4.0  2.0  1.0  3.0;
              3.0  1.0  2.0  2.0  4.0  3.0  1.0]
    capacity = [10.0, 10.0, 12.0]
    return GAPInstance(3, 7, cost, weight, capacity)
end

# Root dual bound is 63
# Optimal value is 63
function gap_small_feasible()
    # 2 machines, 7 tasks — small feasible instance.
    cost = [5.0 8.0 14.0 20.0 5.0 4.0 13.0;
            18.0 14.0 15.0 16.0 3.0 8.0 19.0]
    weight = [1.0 1.0 1.0 5.0 2.0 1.0 4.0;
              5.0 3.0 4.0 1.0 4.0 1.0 1.0]
    capacity = [11.0, 14.0]
    return GAPInstance(2, 7, cost, weight, capacity)
end

# Root dual bound is 70.33333
# Optimal solution is 75
function gap_small_feasible2()
    return GAPInstance(
      2, 7,
      [8 5 11 21 6 5 19;
       1 12 11 12 14 8 5],
      [2 3 3 1 2 1 1;
       5 1 1 3 1 5 4],
      [5, 8]
  )
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Benchmark instance table (Pigatti et al. 2005, Table 1)
# ────────────────────────────────────────────────────────────────────────────────────────

# (class, agents, jobs, expected_dual_bound)
# Subset of Pigatti et al. (2005) Table 1 — tractable at the root node.
# Instance name (letter, number of machine, number of items, Root LB, Optimal solution)
const BENCHMARK_INSTANCES = [
    #('C', 5,  100, 1930),
    #('C', 10, 100, 1400),
    ('C', 20, 100, 1242),
    #('E', 10, 100, 11568),
    #('E', 20, 100, 8431),
]

# ────────────────────────────────────────────────────────────────────────────────────────
# CG e2e tests
# ────────────────────────────────────────────────────────────────────────────────────────

function test_gap_column_generation_converges()
    @testset "[gap] column generation converges (2 machines, 7 tasks)" begin
        inst = gap_small_feasible()
        ctx = build_gap_context(inst)

        output = run_column_generation(ctx)

        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 63.0) <= 1e-4
    end
end

function test_gap_column_generation_converges2()
    @testset "[gap] column generation converges (2 machines, 7 tasks, instance 2)" begin
        inst = gap_small_feasible2()
        ctx = build_gap_context(inst)

        output = run_column_generation(ctx)

        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 70.33333) <= 1e-4
    end
end

function test_gap_benchmark_instances()
    for (class, agents, jobs, expected_bound) in BENCHMARK_INSTANCES
        label = "$(uppercase(class)) $(agents)×$(jobs)"
        @testset "[gap][$(label)] dual bound ≈ $(expected_bound)" begin
            filepath = get_gap_instance_path(class, agents, jobs)
            inst = parse_gap_file(filepath)
            ctx = build_gap_context(inst)

            output = run_column_generation(ctx)

            @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
            @test abs(output.incumbent_dual_bound - expected_bound) <= 1.0
        end
    end
end

function test_gap_infeasible_master()
    @testset "[gap] infeasible master (3 machines, 30 jobs, max 27 coverable)" begin
        inst = gap_infeasible_master()
        ctx  = build_gap_context(inst)
        output = run_column_generation(ctx)
        # Phase0 converges with art vars → Phase1 confirms infeasibility
        @test output.status == master_infeasible
    end
end

function test_gap_infeasible_subproblem()
    @testset "[gap] infeasible subproblem (machine 1 capacity = -1)" begin
        inst = gap_infeasible_subproblem()
        ctx = build_gap_context(inst)
        output = run_column_generation(ctx)
        @test output.status == subproblem_infeasible
    end
end

function test_gap_maximization()
    @testset "[gap] maximization converges (2 machines, 7 tasks, negated costs)" begin
        inst = gap_maximization()
        ctx = build_gap_context_max(inst)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
    end
end

function test_gap_two_identical_machines()
    @testset "[gap] two identical machines (symmetric degeneracy)" begin
        inst = gap_two_identical_machines()
        ctx = build_gap_context(inst)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 32.0) <= 1e-4
    end
end

function test_gap_wentges_smoothing()
    @testset "[gap] Wentges smoothing converges (2 machines, 7 tasks, α=0.5)" begin
        inst = gap_small_feasible2()
        ctx = build_gap_context(inst; smoothing_alpha=0.5)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 70.33333) <= 1e-4
    end
end

function test_gap_with_penalty()
    @testset "[gap] penalty unassignment (3 machines, 30 jobs, penalty=1000)" begin
        gap = gap_infeasible_master()
        penalty = fill(1000.0, gap.n_tasks)
        inst = GAPWithPenaltyInstance(gap, penalty)
        ctx = build_gap_with_penalty_context(inst)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        # At least 3 tasks must be unassigned → penalty ≥ 3000
        @test output.master_lp_obj >= 3000.0 - 1e-4
    end
end

function test_gap_shifted_bounds()
    @testset "[gap] shifted formulation z ∈ {1,2} matches standard bounds" begin
        inst = gap_small_feasible()
        shifted_ctx = build_gap_shifted_context(inst)
        shifted_out = run_column_generation(shifted_ctx)
        @test shifted_out.status == optimal
        @test abs(shifted_out.master_lp_obj - shifted_out.incumbent_dual_bound) <= 1e-4
        @test abs(shifted_out.incumbent_dual_bound - 63.0) <= 1e-4
    end
end

function test_gap_with_penalty2()
    @testset "[gap] penalty unassignment (2 machines, 7 tasks, penalty=10)" begin
        gap = gap_small_feasible2()
        penalty = fill(10.0, gap.n_tasks)
        inst = GAPWithPenaltyInstance(gap, penalty)
        ctx = build_gap_with_penalty_context(inst)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 51.0) <= 1e-4
    end
end

function test_gap_fixed_master_cost()
    @testset "[gap] fixed master cost shifts bounds (2 machines, 7 tasks)" begin
        inst = gap_small_feasible()
        fixed_cost = 100.0
        ctx = build_gap_context_with_fixed_cost(inst, fixed_cost)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - (63.0 + fixed_cost)) <= 1e-4
    end
end


# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function test_column_generation_e2e()
    test_gap_column_generation_converges()
    test_gap_column_generation_converges2()
    #test_gap_benchmark_instances()
    test_gap_infeasible_master()
    test_gap_infeasible_subproblem()
    test_gap_maximization()
    test_gap_two_identical_machines()
    test_gap_with_penalty()
    test_gap_with_penalty2()
    test_gap_shifted_bounds()
    test_gap_fixed_master_cost()
    test_gap_wentges_smoothing()
end
