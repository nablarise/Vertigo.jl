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

function test_dml_silent()
    @testset "[dml] silent is chosen per master and subproblem" begin
        silent(m) = MOI.get(m, MOI.Silent())

        # Default: everything is silent.
        model = _build_gap_jump_model()
        set_optimizer(model, HiGHS.Optimizer)
        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end
        @test silent(decomp.master_model)
        @test all(silent, values(sp_models(decomp)))

        # Explicit choice per model; one rule is enough to decide.
        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k, silent=(k == 2))
            assign[_] => master(silent=false)
            knapsack[k] => subproblem(k)
        end
        @test !silent(decomp.master_model)
        @test silent(sp_models(decomp)[sp_map[2]])
        @test !silent(sp_models(decomp)[sp_map[1]])

        # Rules assigning to the same model must agree.
        @test_throws ErrorException @dantzig_wolfe model begin
            x[k, _] => subproblem(k, silent=true)
            assign[_] => master()
            knapsack[k] => subproblem(k, silent=false)
        end
    end
end

function test_dml_user_overriden_subproblem()
    @testset "[dml] user-overridden subproblem has no model" begin
        model = _build_gap_jump_model()
        set_optimizer(model, HiGHS.Optimizer)
        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k, user_overriden=(k == 1))
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end

        @test sp_model(decomp, sp_map[1]) === nothing
        @test sp_model(decomp, sp_map[2]) !== nothing

        # The user-overridden subproblem is still part of the decomposition: its
        # variables are numbered 1..7 and appear in the coupling rows.
        sp_vars = subproblem_variables(decomp, sp_map[1])
        @test sort(getfield.(sp_vars, :value)) == collect(1:7)
        @test length(subproblem_variables(decomp, sp_map[2])) == 7
        @test length(coupling_constraints(decomp)) == 7
        x = model[:x]
        for t in 1:7
            sp_id, vi = mapped_subproblem_var(decomp, x[1, t])
            @test sp_id == sp_map[1]
            @test vi in sp_vars
            @test mapped_original_var(decomp, sp_id, vi) == x[1, t]
            @test !isempty(coupling_coefficients(decomp, sp_id, vi))
        end

        # Container syntax: options are written once per container.
        decomp, sp_map = @dantzig_wolfe model begin
            master() => (constraints = assign[_])
            subproblem(k, user_overriden=true) => (
                variables = [x[k, _]],
                constraints = knapsack[k]
            )
        end
        @test all(isnothing, values(sp_models(decomp)))
        @test length(coupling_constraints(decomp)) == 7

        # Column generation refuses to start until every pricer is installed.
        @test_throws "has no pricer" ColGenWorkspace(decomp, ColGenConfig())

        # Rules assigning to the same subproblem must agree.
        @test_throws ErrorException @dantzig_wolfe model begin
            x[k, _] => subproblem(k, user_overriden=true)
            assign[_] => master()
            knapsack[k] => subproblem(k, user_overriden=false)
        end
    end
end

function test_dml_annotation_options()
    @testset "[dml] annotation options land in their own fields" begin
        sp = dantzig_wolfe_subproblem(3; user_overriden=true)
        @test sp.id == 3
        @test sp.user_overriden === true
        @test sp.silent === nothing

        sp = dantzig_wolfe_subproblem(:a; silent=false)
        @test sp.id == :a
        @test sp.user_overriden === nothing
        @test sp.silent === false

        sp = dantzig_wolfe_subproblem(1; silent=true, user_overriden=false)
        @test (sp.user_overriden, sp.silent) == (false, true)

        @test dantzig_wolfe_master().silent === nothing
        @test dantzig_wolfe_master(silent=false).silent === false

        # No positional constructor: swapping two `Union{Nothing,Bool}` goes unnoticed.
        @test_throws MethodError Vertigo.DML.SubproblemAnnotation(1, true, false)
        @test_throws MethodError Vertigo.DML.MasterAnnotation(true)
    end
end

# `_replace_vars_in_func` once splatted every term into `AffExpr(constant, terms...)`; Julia
# compiles a new method per argument count and the cost grows far faster than the size, so a
# subproblem with a few thousand variables took tens of seconds to decompose.
function test_dml_large_subproblem_objective()
    @testset "[dml] a subproblem with thousands of variables decomposes quickly" begin
        function build(n)
            model = Model()
            @variable(model, y[1:n], Bin)
            @constraint(model, cover, sum(y) >= 1)
            @objective(model, Min, sum(i * y[i] for i in 1:n))
            return model
        end
        ann(::Val{:y}, i) = dantzig_wolfe_subproblem(1)
        ann(::Val{:cover}) = dantzig_wolfe_master()

        dantzig_wolfe_decomposition(build(10), ann)   # compile everything but the size effect
        n = 2000
        elapsed = @elapsed decomp, _ = dantzig_wolfe_decomposition(build(n), ann)
        @test length(subproblem_variables(decomp, PricingSubproblemId(1))) == n
        @test elapsed < 10   # tens of seconds with the splat; about a second without
    end
end

function test_dml()
    test_dml_user_overriden_subproblem()
    test_dml_macro_basic_gap()
    test_dml_silent()
    test_dml_annotation_function()
    test_dml_pure_master_variables()
    test_dml_annotation_options()
    test_dml_large_subproblem_objective()
end
