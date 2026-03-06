# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function _build_gap_jump_model()
    cost = [5.0 8.0 14.0 20.0 5.0 4.0 13.0;
            18.0 14.0 15.0 16.0 3.0 8.0 19.0]
    weight = [1.0 1.0 1.0 5.0 2.0 1.0 4.0;
              5.0 3.0 4.0 1.0 4.0 1.0 1.0]
    capacity = [11.0, 14.0]
    K = 1:2
    T = 1:7

    model = Model()
    @variable(model, x[k in K, t in T], Bin)
    @constraint(
        model, assign[t in T],
        sum(x[k, t] for k in K) == 1
    )
    @constraint(
        model, knapsack[k in K],
        sum(weight[k, t] * x[k, t] for t in T) <= capacity[k]
    )
    @objective(
        model, Min,
        sum(cost[k, t] * x[k, t] for k in K, t in T)
    )
    return model
end

function test_dml_macro_basic_gap()
    @testset "[dml] @dantzig_wolfe basic GAP structure" begin
        model = _build_gap_jump_model()

        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end

        # 2 subproblems
        sp_ids = collect(subproblem_ids(decomp))
        @test length(sp_ids) == 2

        # Each subproblem has 7 variables
        for sp_id in sp_ids
            @test length(subproblem_variables(decomp, sp_id)) == 7
        end

        # 7 coupling constraints (assignment)
        @test length(coupling_constraints(decomp)) == 7

        # sp_map maps 1 and 2 to PricingSubproblemId
        @test haskey(sp_map, 1)
        @test haskey(sp_map, 2)

        # Minimization
        @test is_minimization(decomp)

        # Convexity bounds (1.0, 1.0)
        for sp_id in sp_ids
            lb, ub = convexity_bounds(decomp, sp_id)
            @test lb == 1.0
            @test ub == 1.0
        end

        # No pure master variables
        @test isempty(pure_master_variables(decomp))
    end
end

function test_dml_annotation_function()
    @testset "[dml] dantzig_wolfe_decomposition with function" begin
        model = _build_gap_jump_model()

        ann(::Val{:x}, k, _) = dantzig_wolfe_subproblem(k)
        ann(::Val{:assign}, _) = dantzig_wolfe_master()
        ann(::Val{:knapsack}, k) = dantzig_wolfe_subproblem(k)

        decomp, sp_map = dantzig_wolfe_decomposition(model, ann)

        sp_ids = collect(subproblem_ids(decomp))
        @test length(sp_ids) == 2

        for sp_id in sp_ids
            @test length(subproblem_variables(decomp, sp_id)) == 7
        end

        @test length(coupling_constraints(decomp)) == 7
    end
end

function test_dml_pure_master_variables()
    @testset "[dml] pure master variables in GAP with penalty" begin
        cost = [8.0 5.0 11.0 21.0 6.0 5.0 19.0;
                1.0 12.0 11.0 12.0 14.0 8.0 5.0]
        weight = [2.0 3.0 3.0 1.0 2.0 1.0 1.0;
                  5.0 1.0 1.0 3.0 1.0 5.0 4.0]
        capacity = [5.0, 8.0]
        penalty = 10.0
        K = 1:2
        T = 1:7

        model = Model()
        @variable(model, x[k in K, t in T], Bin)
        @variable(model, u[t in T], Bin)
        @constraint(
            model, assign[t in T],
            sum(x[k, t] for k in K) + u[t] == 1
        )
        @constraint(
            model, knapsack[k in K],
            sum(weight[k, t] * x[k, t] for t in T) <= capacity[k]
        )
        @objective(
            model, Min,
            sum(cost[k, t] * x[k, t] for k in K, t in T) +
            sum(penalty * u[t] for t in T)
        )

        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            u[_] => master()
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end

        sp_ids = collect(subproblem_ids(decomp))
        @test length(sp_ids) == 2

        pm_vars = pure_master_variables(decomp)
        @test length(pm_vars) == 7

        for pmv in pm_vars
            @test pure_master_cost(decomp, pmv) == penalty
            lb, ub = pure_master_bounds(decomp, pmv)
            @test lb == 0.0
            @test ub == 1.0
            @test Vertigo.Reformulation.pure_master_is_integer(decomp, pmv)
        end
    end
end

function test_dml()
    test_dml_macro_basic_gap()
    test_dml_annotation_function()
    test_dml_pure_master_variables()
end
