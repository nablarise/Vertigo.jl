# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

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
        @test length(subproblem_variables(decomp, sp_ids[1])) == 7
        @test length(subproblem_variables(decomp, sp_ids[2])) == 7

        # 7 coupling constraints (assignment)
        @test length(coupling_constraints(decomp)) == 7

        # sp_map maps 1 and 2 to PricingSubproblemId
        @test haskey(sp_map, 1)
        @test haskey(sp_map, 2)

        # Minimization
        @test is_minimization(decomp)

        # Convexity bounds (1.0, 1.0)
        lb1, ub1 = convexity_bounds(decomp, sp_ids[1])
        @test lb1 ≈ 1.0
        @test ub1 ≈ 1.0
        lb2, ub2 = convexity_bounds(decomp, sp_ids[2])
        @test lb2 ≈ 1.0
        @test ub2 ≈ 1.0

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

        @test length(subproblem_variables(decomp, sp_ids[1])) == 7
        @test length(subproblem_variables(decomp, sp_ids[2])) == 7

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

        @test all(
            pmv -> pure_master_cost(decomp, pmv) ≈ penalty,
            pm_vars
        )
        @test all(pm_vars) do pmv
            lb, ub = pure_master_bounds(decomp, pmv)
            lb ≈ 0.0 && ub ≈ 1.0
        end
        @test all(
            pmv -> Vertigo.Reformulation.pure_master_is_integer(
                decomp, pmv
            ),
            pm_vars
        )
    end
end

# The decomposition records the bounds the master keeps: a missing lower bound
# is -Inf, not 0, and a fixing is both bounds.
function test_dml_pure_master_missing_bounds()
    # (name, bounds of `y`, expected lower and upper bounds)
    cases = [
        ("free", _ -> nothing, -Inf, Inf),
        ("upper bound only", y -> set_upper_bound(y, 4), -Inf, 4.0),
        ("lower bound only", y -> set_lower_bound(y, 2), 2.0, Inf),
        ("fixed at -2", y -> fix(y, -2), -2.0, -2.0),
        ("fixed at 0", y -> fix(y, 0), 0.0, 0.0),
        ("fixed at 4", y -> fix(y, 4), 4.0, 4.0),
    ]
    for (name, set_bounds!, expected_lb, expected_ub) in cases
        @testset "[dml] pure master variable bounds, $name" begin
            decomp = build_pure_master_y_decomp(set_bounds!)
            lb, ub = pure_master_bounds(decomp, only(pure_master_variables(decomp)))
            @test lb ≈ expected_lb atol=1e-10
            @test ub ≈ expected_ub atol=1e-10
        end
    end
end

function test_dml_solver_parameters()
    @testset "[dml] models are silent and take parameters after decomposition" begin
        silent(m) = MOI.get(m, MOI.Silent())
        model = _build_gap_jump_model()
        set_optimizer(model, HiGHS.Optimizer)
        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end
        @test silent(master_model(decomp))
        @test all(silent, values(sp_models(decomp)))

        MOI.set(sp_model(decomp, sp_map[1]), MOI.Silent(), false)
        @test !silent(sp_model(decomp, sp_map[1]))
        @test silent(sp_model(decomp, sp_map[2]))
        @test silent(master_model(decomp))
    end
end

function test_dml_containers_real_indices()
    @testset "[dml] every JuMP container is decomposed by its real indices" begin
        # x[k, t]: two subproblems "a" and "b" over tasks 3:5, linked by cover[t].
        # The decomposition must be the same whatever container JuMP stores them in.
        for container in (
            JuMP.Containers.DenseAxisArray, JuMP.Containers.SparseAxisArray, Dict
        )
            model = Model()
            K, T = ["a", "b"], 3:5
            x = @variable(model, [k in K, t in T], base_name = "x", container = container)
            model[:x] = x
            model[:cover] = @constraint(
                model, [t in T], sum(x[k, t] for k in K) == 1, container = container
            )
            @constraint(
                model, cap[k in K], sum(x[k, t] for t in T) <= 2, container = Dict
            )
            @objective(model, Min, sum(x[k, t] for k in K, t in T))
            decomp, sp_map = @dantzig_wolfe model begin
                x[k, _] => subproblem(k)
                cover[_] => master()
                cap[k] => subproblem(k)
            end
            @test sort(collect(keys(sp_map))) == ["a", "b"]
            for id in subproblem_ids(decomp)
                @test length(subproblem_variables(decomp, id)) == 3
            end
            @test length(coupling_constraints(decomp)) == 3
        end
    end
end

# x[1] is in subproblem 1, x[2] in subproblem 2 and y in the master. A
# subproblem constraint over a variable it does not own would lose that term.
function test_dml_subproblem_constraint_foreign_variable()
    @testset "[dml] subproblem constraint with a variable of another subproblem" begin
        model = Model()
        @variable(model, x[1:2], Bin)
        @constraint(model, row, x[1] + x[2] <= 0)
        @test_throws ["Constraint row ", "subproblem 1", "x[2], a variable of subproblem 2"] (
            @dantzig_wolfe model begin
                x[i] => subproblem(i)
                row => subproblem(1)
            end
        )
    end
    @testset "[dml] indexed subproblem constraint with a variable of another subproblem" begin
        model = Model()
        @variable(model, x[1:2], Bin)
        @constraint(model, cap[k in 1:2], x[k] + (k == 1 ? x[2] : 0) <= 1)
        @test_throws ["Constraint cap[1] ", "x[2], a variable of subproblem 2"] (
            @dantzig_wolfe model begin
                x[i] => subproblem(i)
                cap[k] => subproblem(k)
            end
        )
    end
    @testset "[dml] subproblem constraint with a master variable" begin
        model = Model()
        @variable(model, x[1:2], Bin)
        @variable(model, y)
        @constraint(model, row, x[1] + y <= 1)
        @test_throws ["Constraint row ", "y, a variable of the master"] (
            @dantzig_wolfe model begin
                x[i] => subproblem(i)
                y => master()
                row => subproblem(1)
            end
        )
    end
    @testset "[dml] the same constraint in the master keeps both variables" begin
        model = Model()
        @variable(model, x[1:2], Bin)
        @constraint(model, row, x[1] + x[2] <= 0)
        decomp, _ = @dantzig_wolfe model begin
            x[i] => subproblem(i)
            row => master()
        end
        @test length(coupling_constraints(decomp)) == 1
        for sp_id in subproblem_ids(decomp)
            sp_var = only(subproblem_variables(decomp, sp_id))
            entry = only(coupling_coefficients(decomp, sp_id, sp_var))
            @test entry.coefficient ≈ 1.0 atol=1e-10
        end
    end
end

function test_dml_unassigned_constraints()
    @testset "[dml] an anonymous constraint is not silently left out" begin
        model = _build_gap_jump_model()
        x = model[:x]
        @constraint(model, x[1, 1] + x[2, 1] <= 0)
        @test_throws ["Constraint ", "assigned to neither the master nor a subproblem"] (
            @dantzig_wolfe model begin
                x[k, _] => subproblem(k)
                assign[_] => master()
                knapsack[k] => subproblem(k)
            end
        )
    end
    @testset "[dml] a named constraint assigned to no model is not left out" begin
        model = _build_gap_jump_model()
        ann(::Val{:x}, k, _) = dantzig_wolfe_subproblem(k)
        ann(::Val{:assign}, _) = dantzig_wolfe_master()
        ann(::Val{:knapsack}, k) = nothing
        @test_throws ["Constraint knapsack[1]", "assigned to neither the master nor a subproblem"] (
            dantzig_wolfe_decomposition(model, ann)
        )
    end
end

function test_dml_ranged_master_constraint()
    @testset "[dml] a ranged master constraint is rejected" begin
        model = Model()
        @variable(model, x[1:2], Bin)
        @constraint(model, r, 1 <= x[1] + x[2] <= 2)
        @test_throws ["Master constraint r ", "two named constraints"] (
            @dantzig_wolfe model begin
                x[i] => subproblem(i)
                r => master()
            end
        )
    end
    @testset "[dml] a ranged constraint written as two master constraints" begin
        model = Model()
        @variable(model, x[1:2], Bin)
        @constraint(model, r_lb, x[1] + x[2] >= 1)
        @constraint(model, r_ub, x[1] + x[2] <= 2)
        decomp, _ = @dantzig_wolfe model begin
            x[i] => subproblem(i)
            r_lb => master()
            r_ub => master()
        end
        @test length(coupling_constraints(decomp)) == 2
    end
end

function test_dml()
    test_dml_macro_basic_gap()
    test_dml_annotation_function()
    test_dml_containers_real_indices()
    test_dml_pure_master_variables()
    test_dml_pure_master_missing_bounds()
    test_dml_solver_parameters()
    test_dml_subproblem_constraint_foreign_variable()
    test_dml_unassigned_constraints()
    test_dml_ranged_master_constraint()
end
