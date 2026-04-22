# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

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

# Root dual bound is 32
# Optimal value is 32
# Same as gap_two_identical_machines but with 2 machine types:
#   type 1 = identical pair (multiplicity 2), type 2 = machine 3
function gap_identical_machines()
    cost = [5.0  8.0  3.0  12.0  7.0  4.0  9.0;
            8.0  6.0  10.0  4.0  11.0  7.0  3.0]
    weight = [2.0  3.0  1.0  4.0  2.0  1.0  3.0;
              3.0  1.0  2.0  2.0  4.0  3.0  1.0]
    capacity = [10.0, 12.0]
    return GAPInstanceWithIdenticalMachines(
        2, 7, cost, weight, capacity, [2, 1]
    )
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

function test_gap_column_generation_converges(; smoothing_alpha=0.0)
    @testset "[gap] column generation converges (2 machines, 7 tasks)" begin
        expected_root_dual_bound = 63.0
        inst = gap_small_feasible()
        ws = build_gap_context(inst; smoothing_alpha)
        output = run_column_generation(ws)
        @test output.status == optimal
        @test output.master_lp_obj ≈ output.incumbent_dual_bound atol=1e-4
        @test output.incumbent_dual_bound ≈ expected_root_dual_bound atol=1e-4
    end
end

function test_gap_column_generation_converges2(; smoothing_alpha=0.0)
    @testset "[gap] column generation converges (2 machines, 7 tasks, instance 2)" begin
        expected_root_dual_bound = 70.33333
        inst = gap_small_feasible2()
        ws = build_gap_context(inst; smoothing_alpha)
        output = run_column_generation(ws)
        @test output.status == optimal
        @test output.master_lp_obj ≈ output.incumbent_dual_bound atol=1e-4
        @test output.incumbent_dual_bound ≈ expected_root_dual_bound atol=1e-4
    end
end

function test_gap_benchmark_instances(; smoothing_alpha=0.0)
    for (class, agents, jobs, expected_bound) in BENCHMARK_INSTANCES
        label = "$(uppercase(class)) $(agents)×$(jobs)"
        @testset "[gap][$(label)] dual bound ≈ $(expected_bound)" begin
            filepath = get_gap_instance_path(class, agents, jobs)
            inst = parse_gap_file(filepath)
            ws = build_gap_context(inst; smoothing_alpha)

            output = run_column_generation(ws)

            @test output.master_lp_obj ≈ output.incumbent_dual_bound atol=1e-4
            @test output.incumbent_dual_bound ≈ expected_bound atol=1.0
        end
    end
end

function test_gap_infeasible_master(; smoothing_alpha=0.0)
    @testset "[gap] infeasible master (3 machines, 30 jobs, max 27 coverable)" begin
        inst = gap_infeasible_master()
        ws  = build_gap_context(inst; smoothing_alpha)
        output = run_column_generation(ws)
        # Phase0 converges with art vars → Phase1 confirms infeasibility
        @test output.status == master_infeasible
    end
end

function test_gap_infeasible_subproblem(; smoothing_alpha=0.0)
    @testset "[gap] infeasible subproblem (machine 1 capacity = -1)" begin
        inst = gap_infeasible_subproblem()
        ws = build_gap_context(inst; smoothing_alpha)
        output = run_column_generation(ws)
        @test output.status == subproblem_infeasible
    end
end

function test_gap_maximization(; smoothing_alpha=0.0)
    @testset "[gap] maximization converges (2 machines, 7 tasks, negated costs)" begin
        inst = gap_maximization()
        ws = build_gap_context_max(inst; smoothing_alpha)
        output = run_column_generation(ws)
        @test output.status == optimal
        @test output.master_lp_obj ≈ output.incumbent_dual_bound atol=1e-4
    end
end

function test_gap_two_identical_machines(; smoothing_alpha=0.0)
    @testset "[gap] two identical machines (symmetric degeneracy)" begin
        expected_root_dual_bound = 32.0
        inst = gap_two_identical_machines()
        ws = build_gap_context(inst; smoothing_alpha)
        output = run_column_generation(ws)
        @test output.status == optimal
        @test output.master_lp_obj ≈ output.incumbent_dual_bound atol=1e-4
        @test output.incumbent_dual_bound ≈ expected_root_dual_bound atol=1e-4
    end
end

function test_gap_identical_machines(; smoothing_alpha=0.0)
    @testset "[gap] identical machines with multiplicity (2 types, 7 tasks)" begin
        expected_root_dual_bound = 32.0
        inst = gap_identical_machines()
        ws = build_gap_identical_context(inst; smoothing_alpha)
        output = run_column_generation(ws)
        @test output.status == optimal
        @test output.master_lp_obj ≈ output.incumbent_dual_bound atol=1e-4
        @test output.incumbent_dual_bound ≈ expected_root_dual_bound atol=1e-4
    end
end

function test_gap_shifted_bounds(; smoothing_alpha=0.0)
    @testset "[gap] shifted formulation z ∈ {1,2} matches standard bounds" begin
        expected_root_dual_bound = 63.0
        inst = gap_small_feasible()
        shifted_ws = build_gap_shifted_context(inst; smoothing_alpha)
        shifted_out = run_column_generation(shifted_ws)
        @test shifted_out.status == optimal
        @test shifted_out.master_lp_obj ≈ shifted_out.incumbent_dual_bound atol=1e-4
        @test shifted_out.incumbent_dual_bound ≈ expected_root_dual_bound atol=1e-4
    end
end

function test_gap_with_penalty(; smoothing_alpha=0.0)
    @testset "[gap] penalty unassignment (2 machines, 7 tasks, penalty=10)" begin
        gap = gap_small_feasible2()
        penalty = fill(10.0, gap.n_tasks)
        inst = GAPWithPenaltyInstance(gap, penalty)
        ws = build_gap_with_penalty_context(inst; smoothing_alpha)
        output = run_column_generation(ws)
        @test output.status == optimal
        @test output.master_lp_obj ≈ output.incumbent_dual_bound atol=1e-4
        expected_root_dual_bound = 51.0
        @test output.incumbent_dual_bound ≈ expected_root_dual_bound atol=1e-4
    end
end

function test_gap_with_penalty_cardinality(; smoothing_alpha=0.0)
    @testset "[gap] penalty + cardinality (2 machines, 7 tasks, penalty=5, max 3 unassigned)" begin
        gap = gap_small_feasible2()
        penalty = fill(5.0, gap.n_tasks)
        inst = GAPWithPenaltyCardInstance(gap, penalty, 3)
        ws = build_gap_with_penalty_card_context(inst; smoothing_alpha)
        output = run_column_generation(ws)
        @test output.status == optimal
        @test output.master_lp_obj ≈ output.incumbent_dual_bound atol=1e-4
        expected_root_dual_bound = 37.0
        @test output.incumbent_dual_bound ≈ expected_root_dual_bound atol=1e-4
    end
end

function test_gap_fixed_master_cost(; smoothing_alpha=0.0)
    @testset "[gap] fixed master cost shifts bounds (2 machines, 7 tasks)" begin
        inst = gap_small_feasible()
        fixed_cost = 100.0
        ws = build_gap_context_with_fixed_cost(inst, fixed_cost; smoothing_alpha)
        output = run_column_generation(ws)
        @test output.status == optimal
        @test output.master_lp_obj ≈ output.incumbent_dual_bound atol=1e-4
        expected_root_dual_bound = 163.0
        @test output.incumbent_dual_bound ≈ expected_root_dual_bound atol=1e-4
    end
end

function test_gap_ip_pruned()
    @testset "[gap] ip pruned when cutoff below LP relaxation" begin
        inst = gap_small_feasible2()
        ws = build_gap_context(inst)
        # Set IP cutoff below LP relaxation (≈70.33)
        # Direct field access: no public API to set IP cutoff.
        raw = ws.inner
        raw.ip_primal_bound = 60.0
        output = run_column_generation(ws)
        @test output.status == ip_pruned
        @test !isnothing(output.master_lp_obj)
        @test !isnothing(output.incumbent_dual_bound)
    end
end

function test_gap_dual_bound_with_pure_master(; smoothing_alpha=0.0)
    @testset "[gap] dual bound with pure master variable (pure_master_contrib bug)" begin
        inst = random_gap_instance(1, 2; seed=123)
        ws = build_gap_with_pure_master_context(inst; smoothing_alpha)

        output = run_column_generation(ws)

        @test output.status == optimal
        @test !isnothing(output.incumbent_dual_bound)
        @test !isnothing(output.master_lp_obj)

        # At convergence, dual bound must equal master LP objective
        # (strong duality). The pure_master_contrib double-counting bug makes
        # dual_bound < master_lp_obj for minimization (invalid).
        @test output.incumbent_dual_bound ≈ output.master_lp_obj atol=1e-4
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function test_column_generation_e2e()
    test_gap_ip_pruned()
    for smoothing_alpha in (0.0, 0.5)
        test_gap_column_generation_converges(; smoothing_alpha)
        test_gap_column_generation_converges2(; smoothing_alpha)
        #test_gap_benchmark_instances(; smoothing_alpha)
        test_gap_infeasible_master(; smoothing_alpha)
        test_gap_infeasible_subproblem(; smoothing_alpha)
        test_gap_maximization(; smoothing_alpha)
        test_gap_two_identical_machines(; smoothing_alpha)
        test_gap_identical_machines(; smoothing_alpha)
        test_gap_with_penalty(; smoothing_alpha)
        test_gap_with_penalty_cardinality(; smoothing_alpha)
        test_gap_shifted_bounds(; smoothing_alpha)
        test_gap_fixed_master_cost(; smoothing_alpha)
        test_gap_dual_bound_with_pure_master(; smoothing_alpha)
    end
end
