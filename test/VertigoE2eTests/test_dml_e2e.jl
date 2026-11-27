# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

function _build_colgen_context_from_dml(
    decomp, sp_map;
    smoothing_alpha::Float64=0.0
)
    config = ColGenConfig(smoothing_alpha=smoothing_alpha)
    workspace = ColGenWorkspace(decomp, config)
    return ColGenLoggerWorkspace(workspace)
end

function test_dml_gap_basic(; smoothing_alpha=0.0)
    @testset "[dml] basic GAP (2m, 7t) dual bound ≈ 63.0" begin
        cost = [5.0 8.0 14.0 20.0 5.0 4.0 13.0;
                18.0 14.0 15.0 16.0 3.0 8.0 19.0]
        weight = [1.0 1.0 1.0 5.0 2.0 1.0 4.0;
                  5.0 3.0 4.0 1.0 4.0 1.0 1.0]
        capacity = [11.0, 14.0]
        K = 1:2; T = 1:7

        model = Model()
        @variable(model, x[k in K, t in T], Bin)
        @constraint(model, assign[t in T],
            sum(x[k, t] for k in K) == 1)
        @constraint(model, knapsack[k in K],
            sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
        @objective(model, Min,
            sum(cost[k, t] * x[k, t] for k in K, t in T))

        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end

        _set_optimizer_on_models!(decomp, HiGHS.Optimizer)

        ws = _build_colgen_context_from_dml(
            decomp, sp_map; smoothing_alpha
        )
        output = run_column_generation(ws)
        @test output.status == optimal
        @test abs(output.incumbent_dual_bound - 63.0) <= 1e-4
    end
end

function _set_optimizer_on_models!(decomp, optimizer)
    m = Vertigo.Reformulation.master_model(decomp)
    MOI.set(m, MOI.Silent(), true)
    for (_, sp) in Vertigo.Reformulation.sp_models(decomp)
        MOI.set(sp, MOI.Silent(), true)
    end
end

function _build_dml_gap(cost, weight, capacity; sense=JuMP.MIN_SENSE)
    K = 1:size(cost, 1)
    T = 1:size(cost, 2)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[k in K, t in T], Bin)
    @constraint(model, assign[t in T],
        sum(x[k, t] for k in K) == 1)
    @constraint(model, knapsack[k in K],
        sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
    @objective(model, sense,
        sum(cost[k, t] * x[k, t] for k in K, t in T))

    return @dantzig_wolfe model begin
        x[k, _] => subproblem(k)
        assign[_] => master()
        knapsack[k] => subproblem(k)
    end
end

function test_dml_gap_converges(; smoothing_alpha=0.0)
    @testset "[dml] GAP converges (2m, 7t) dual ≈ 63.0" begin
        cost = [5.0 8.0 14.0 20.0 5.0 4.0 13.0;
                18.0 14.0 15.0 16.0 3.0 8.0 19.0]
        weight = [1.0 1.0 1.0 5.0 2.0 1.0 4.0;
                  5.0 3.0 4.0 1.0 4.0 1.0 1.0]
        capacity = [11.0, 14.0]

        decomp, sp_map = _build_dml_gap(cost, weight, capacity)
        _set_optimizer_on_models!(decomp, HiGHS.Optimizer)

        ws = _build_colgen_context_from_dml(
            decomp, sp_map; smoothing_alpha
        )
        output = run_column_generation(ws)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 63.0) <= 1e-4
    end
end

function test_dml_gap_converges2(; smoothing_alpha=0.0)
    @testset "[dml] GAP instance 2 (2m, 7t) dual ≈ 70.333" begin
        cost = [8.0 5.0 11.0 21.0 6.0 5.0 19.0;
                1.0 12.0 11.0 12.0 14.0 8.0 5.0]
        weight = [2.0 3.0 3.0 1.0 2.0 1.0 1.0;
                  5.0 1.0 1.0 3.0 1.0 5.0 4.0]
        capacity = [5.0, 8.0]

        decomp, sp_map = _build_dml_gap(cost, weight, capacity)
        _set_optimizer_on_models!(decomp, HiGHS.Optimizer)

        ws = _build_colgen_context_from_dml(
            decomp, sp_map; smoothing_alpha
        )
        output = run_column_generation(ws)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 70.33333) <= 1e-4
    end
end

function test_dml_maximization(; smoothing_alpha=0.0)
    @testset "[dml] maximization GAP converges" begin
        cost = -[8.0 5.0 11.0 21.0 6.0 5.0 19.0;
                 1.0 12.0 11.0 12.0 14.0 8.0 5.0]
        weight = [2.0 3.0 3.0 1.0 2.0 1.0 1.0;
                  5.0 1.0 1.0 3.0 1.0 5.0 4.0]
        capacity = [5.0, 8.0]

        decomp, sp_map = _build_dml_gap(
            cost, weight, capacity; sense=JuMP.MAX_SENSE
        )
        _set_optimizer_on_models!(decomp, HiGHS.Optimizer)

        ws = _build_colgen_context_from_dml(
            decomp, sp_map; smoothing_alpha
        )
        output = run_column_generation(ws)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
    end
end

function test_dml_with_penalty(; smoothing_alpha=0.0)
    @testset "[dml] GAP with penalty (2m, 7t, penalty=10) dual ≈ 51.0" begin
        cost = [8.0 5.0 11.0 21.0 6.0 5.0 19.0;
                1.0 12.0 11.0 12.0 14.0 8.0 5.0]
        weight = [2.0 3.0 3.0 1.0 2.0 1.0 1.0;
                  5.0 1.0 1.0 3.0 1.0 5.0 4.0]
        capacity = [5.0, 8.0]
        penalty = 10.0
        K = 1:2; T = 1:7

        model = Model(HiGHS.Optimizer)
        set_silent(model)
        @variable(model, x[k in K, t in T], Bin)
        @variable(model, u[t in T], Bin)
        @constraint(model, assign[t in T],
            sum(x[k, t] for k in K) + u[t] == 1)
        @constraint(model, knapsack[k in K],
            sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
        @objective(model, Min,
            sum(cost[k, t] * x[k, t] for k in K, t in T) +
            sum(penalty * u[t] for t in T))

        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            u[_] => master()
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end
        _set_optimizer_on_models!(decomp, HiGHS.Optimizer)

        ws = _build_colgen_context_from_dml(
            decomp, sp_map; smoothing_alpha
        )
        output = run_column_generation(ws)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 51.0) <= 1e-4
    end
end

function test_dml_with_penalty_cardinality(; smoothing_alpha=0.0)
    @testset "[dml] GAP penalty + cardinality (2m, 7t, p=5, max3) dual ≈ 37.0" begin
        cost = [8.0 5.0 11.0 21.0 6.0 5.0 19.0;
                1.0 12.0 11.0 12.0 14.0 8.0 5.0]
        weight = [2.0 3.0 3.0 1.0 2.0 1.0 1.0;
                  5.0 1.0 1.0 3.0 1.0 5.0 4.0]
        capacity = [5.0, 8.0]
        penalty = 5.0
        max_unassigned = 3
        K = 1:2; T = 1:7

        model = Model(HiGHS.Optimizer)
        set_silent(model)
        @variable(model, x[k in K, t in T], Bin)
        @variable(model, u[t in T], Bin)
        @constraint(model, assign[t in T],
            sum(x[k, t] for k in K) + u[t] == 1)
        @constraint(model, knapsack[k in K],
            sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
        @constraint(model, max_unassign,
            sum(u[t] for t in T) <= max_unassigned)
        @objective(model, Min,
            sum(cost[k, t] * x[k, t] for k in K, t in T) +
            sum(penalty * u[t] for t in T))

        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            u[_] => master()
            assign[_] => master()
            knapsack[k] => subproblem(k)
            max_unassign => master()
        end
        _set_optimizer_on_models!(decomp, HiGHS.Optimizer)

        ws = _build_colgen_context_from_dml(
            decomp, sp_map; smoothing_alpha
        )
        output = run_column_generation(ws)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 37.0) <= 1e-4
    end
end

function test_dml_fixed_master_cost(; smoothing_alpha=0.0)
    @testset "[dml] fixed master cost (2m, 7t, fixed=100) dual ≈ 163.0" begin
        cost = [5.0 8.0 14.0 20.0 5.0 4.0 13.0;
                18.0 14.0 15.0 16.0 3.0 8.0 19.0]
        weight = [1.0 1.0 1.0 5.0 2.0 1.0 4.0;
                  5.0 3.0 4.0 1.0 4.0 1.0 1.0]
        capacity = [11.0, 14.0]
        fixed_cost = 100.0
        K = 1:2; T = 1:7

        model = Model(HiGHS.Optimizer)
        set_silent(model)
        @variable(model, x[k in K, t in T], Bin)
        @constraint(model, assign[t in T],
            sum(x[k, t] for k in K) == 1)
        @constraint(model, knapsack[k in K],
            sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
        @objective(model, Min,
            sum(cost[k, t] * x[k, t] for k in K, t in T) + fixed_cost)

        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end
        _set_optimizer_on_models!(decomp, HiGHS.Optimizer)

        ws = _build_colgen_context_from_dml(
            decomp, sp_map; smoothing_alpha
        )
        output = run_column_generation(ws)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
        @test abs(output.incumbent_dual_bound - 163.0) <= 1e-4
        @test !isnothing(output.ip_incumbent)
        # The GAP optimum 63 (test_dml_gap_basic) plus the constant 100.
        @test abs(output.ip_incumbent.obj_value - 163.0) <= 1e-4
    end
end

# One binary forced to 1, objective 3x + constant: the only solution is worth
# 3 + constant, in either sense.
function test_dml_objective_constant_incumbent()
    for sense in (MOI.MIN_SENSE, MOI.MAX_SENSE), constant in (0.0, 100.0, -100.0)
        @testset "[dml] incumbent counts the objective constant, $sense, $constant" begin
            model = Model(HiGHS.Optimizer)
            set_silent(model)
            @variable(model, x, Bin)
            @constraint(model, must_choose, 1.0 * x >= 1)
            @objective(model, sense, 3x + constant)
            # Each run adds columns to the decomposition: one per run.
            build_decomp() = first(@dantzig_wolfe model begin
                x => subproblem(1)
                must_choose => subproblem(1)
            end)
            expected = 3.0 + constant

            cg = run_col_gen(build_decomp(), ColGenConfig(silent=true))
            @test cg.status == optimal
            @test !isnothing(cg.ip_incumbent)
            @test cg.ip_incumbent.obj_value ≈ expected atol=1e-6

            bp = run_branch_and_price(BranchCutPriceWorkspace(
                build_decomp(), BranchCutPriceConfig(rmp_heuristic=false)
            ))
            @test bp.status == Vertigo.OPTIMAL
            @test bp.incumbent.obj_value ≈ expected atol=1e-6
            @test bp.best_dual_bound ≈ expected atol=1e-6
        end
    end
end

function test_dml_dual_bound_pure_master(; smoothing_alpha=0.0)
    @testset "[dml] dual bound with pure master variable" begin
        cost = [8.0 5.0 11.0 21.0 6.0 5.0 19.0;
                1.0 12.0 11.0 12.0 14.0 8.0 5.0]
        weight = [2.0 3.0 3.0 1.0 2.0 1.0 1.0;
                  5.0 1.0 1.0 3.0 1.0 5.0 4.0]
        capacity = [5.0, 8.0]
        K = 1:2; T = 1:7

        model = Model(HiGHS.Optimizer)
        set_silent(model)
        @variable(model, x[k in K, t in T], Bin)
        @variable(model, 0 <= y <= 1)
        @constraint(model, assign[t in T],
            sum(x[k, t] for k in K) == 1)
        @constraint(model, knapsack[k in K],
            sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
        @constraint(model, link,
            x[1, 1] + y == 1)
        @objective(model, Min,
            sum(cost[k, t] * x[k, t] for k in K, t in T) + 2.0 * y)

        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            y => master()
            assign[_] => master()
            knapsack[k] => subproblem(k)
            link => master()
        end
        _set_optimizer_on_models!(decomp, HiGHS.Optimizer)

        ws = _build_colgen_context_from_dml(
            decomp, sp_map; smoothing_alpha
        )
        output = run_column_generation(ws)
        @test output.status == optimal
        @test output.incumbent_dual_bound ≈ output.master_lp_obj atol=1e-4
    end
end

# GAP of test_dml_dual_bound_pure_master where task 1 is covered by `y`, fixed
# at 1 and of cost 50. `fixed = false` substitutes 1 for `y` instead.
function _build_dml_gap_fixed_pure_master(fixed)
    cost = [8.0 5.0 11.0 21.0 6.0 5.0 19.0;
            1.0 12.0 11.0 12.0 14.0 8.0 5.0]
    weight = [2.0 3.0 3.0 1.0 2.0 1.0 1.0;
              5.0 1.0 1.0 3.0 1.0 5.0 4.0]
    capacity = [5.0, 8.0]
    K = 1:2; T = 1:7

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[k in K, t in T], Bin)
    @constraint(model, knapsack[k in K],
        sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
    if fixed
        @variable(model, y)
        fix(y, 1.0)
        @constraint(model, assign[t in T],
            sum(x[k, t] for k in K) + (t == 1 ? y : 0.0) == 1)
        @objective(model, Min,
            sum(cost[k, t] * x[k, t] for k in K, t in T) + 50.0 * y)
        decomp, _ = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            y => master()
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end
    else
        @constraint(model, assign[t in T],
            sum(x[k, t] for k in K) + (t == 1 ? 1.0 : 0.0) == 1)
        @objective(model, Min,
            sum(cost[k, t] * x[k, t] for k in K, t in T) + 50.0)
        decomp, _ = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end
    end
    return decomp
end

function test_dml_dual_bound_fixed_pure_master()
    @testset "[dml] dual bound with a fixed pure master variable" begin
        fixed = run_col_gen(
            _build_dml_gap_fixed_pure_master(true), ColGenConfig(silent=true)
        )
        substituted = run_col_gen(
            _build_dml_gap_fixed_pure_master(false), ColGenConfig(silent=true)
        )
        @test fixed.status == optimal
        @test substituted.status == optimal
        # HiGHS gives `y == 1` the dual 50 at the optimum: nonzero, so
        # counting it twice would move the bound.
        @test fixed.incumbent_dual_bound ≈ fixed.master_lp_obj atol=1e-4
        @test fixed.incumbent_dual_bound ≈ substituted.incumbent_dual_bound atol=1e-4
        # Compact model solved by HiGHS: LP relaxation 97.75, optimum 110.
        @test 97.75 - 1e-4 <= fixed.incumbent_dual_bound <= 110.0 + 1e-4
    end
end

# 1 <= x[1] + x[2] <= 2 written as two master constraints: the minimum of
# x[1] + x[2] over binaries is 1.
function test_dml_ranged_constraint_as_two_rows()
    @testset "[dml] ranged constraint written as two master constraints" begin
        model = Model(HiGHS.Optimizer)
        set_silent(model)
        @variable(model, x[1:2], Bin)
        @constraint(model, r_lb, x[1] + x[2] >= 1)
        @constraint(model, r_ub, x[1] + x[2] <= 2)
        @objective(model, Min, x[1] + x[2])
        decomp, _ = @dantzig_wolfe model begin
            x[i] => subproblem(i)
            r_lb => master()
            r_ub => master()
        end
        output = run_col_gen(decomp, ColGenConfig(silent=true))
        @test output.status == optimal
        @test output.master_lp_obj ≈ 1.0 atol=1e-6
        @test output.incumbent_dual_bound ≈ 1.0 atol=1e-6
    end
end

# Three items of weight 5 and bins of capacity 10: a bin holds two items, and
# y is the setup variable of a bin.
# Hand computation: every column covers at most two of the three items, so
# the LP needs at least 3 / 2 = 1.5 bins, reached by the three pairs at 1/2.
function _build_dml_bin_packing(nb_bins)
    weight = [5.0, 5.0, 5.0]
    capacity = 10.0
    items = eachindex(weight)
    K = identical(:bins, nb_bins)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[k in K, i in items], Bin)
    @variable(model, y[k in K], Bin)
    @constraint(model, knapsack[k in K],
        sum(weight[i] * x[k, i] for i in items) <= capacity * y[k])
    @constraint(model, cover[i in items], sum(x[k, i] for k in K) >= 1)
    # A single variable: the DML must still give the models an affine objective.
    @objective(model, Min, sum(y[k] for k in K))

    decomp, _ = @dantzig_wolfe model begin
        x[k, _] => subproblem(k)
        y[k] => setup(k)
        knapsack[k] => subproblem(k)
        cover[_] => master()
    end
    return decomp
end

function test_dml_bin_packing_identical(; smoothing_alpha=0.0)
    @testset "[dml] bin packing on identical bins, root bound 1.5" begin
        decomp = _build_dml_bin_packing(3)
        config = ColGenConfig(smoothing_alpha=smoothing_alpha, silent=true)
        output = run_col_gen(decomp, config)
        @test output.status == optimal
        @test output.master_lp_obj ≈ 1.5 atol=1e-6
        @test output.incumbent_dual_bound ≈ 1.5 atol=1e-6
    end
end

function test_dml_bin_packing_too_few_bins()
    @testset "[dml] bin packing with one bin for three items is infeasible" begin
        # One bin holds two of the three items.
        decomp = _build_dml_bin_packing(1)
        output = run_col_gen(decomp, ColGenConfig(silent=true))
        @test output.status == master_infeasible
    end
end

function test_dml_gap_identical_machines(; smoothing_alpha=0.0)
    @testset "[dml] GAP on identical machines (2 types, 7 tasks) dual ≈ 32.0" begin
        inst = gap_identical_machines()
        K = [identical(Symbol(:type, m), inst.multiplicity[m])
             for m in 1:inst.n_machine_types]
        type_of = Dict(k => m for (m, k) in enumerate(K))
        T = 1:inst.n_tasks

        model = Model(HiGHS.Optimizer)
        set_silent(model)
        @variable(model, x[k in K, t in T], Bin)
        @constraint(model, assign[t in T], sum(x[k, t] for k in K) == 1)
        @constraint(model, knapsack[k in K],
            sum(inst.weight[type_of[k], t] * x[k, t] for t in T) <=
            inst.capacity[type_of[k]])
        @objective(model, Min,
            sum(inst.cost[type_of[k], t] * x[k, t] for k in K, t in T))

        decomp, _ = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end
        config = ColGenConfig(smoothing_alpha=smoothing_alpha, silent=true)
        output = run_col_gen(decomp, config)
        @test output.status == optimal
        # Same root bound as the decomposition built by hand,
        # test_gap_identical_machines.
        @test output.incumbent_dual_bound ≈ 32.0 atol=1e-4
    end
end

function test_dml_e2e()
    for smoothing_alpha in (0.0, 0.5)
        test_dml_gap_converges(; smoothing_alpha)
        test_dml_gap_converges2(; smoothing_alpha)
        test_dml_maximization(; smoothing_alpha)
        test_dml_with_penalty(; smoothing_alpha)
        test_dml_with_penalty_cardinality(; smoothing_alpha)
        test_dml_fixed_master_cost(; smoothing_alpha)
        test_dml_dual_bound_pure_master(; smoothing_alpha)
        test_dml_bin_packing_identical(; smoothing_alpha)
        test_dml_gap_identical_machines(; smoothing_alpha)
    end
    test_dml_objective_constant_incumbent()
    test_dml_dual_bound_fixed_pure_master()
    test_dml_ranged_constraint_as_two_rows()
    test_dml_bin_packing_too_few_bins()
end
