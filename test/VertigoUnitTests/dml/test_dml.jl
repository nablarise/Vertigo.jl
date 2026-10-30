# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# The GAP model of `_build_gap_jump_model` (or one with the same names), decomposed by
# machine: the master keeps `assign`, and also `variables` and `constraints`.
function _decompose_gap(
    model;
    variables=(),
    constraints=(),
    master_options=MasterProblemOptions(),
    options=k -> SubproblemOptions(),
)
    x, assign, knapsack = model[:x], model[:assign], model[:knapsack]
    return dantzig_wolfe_decomposition(
        model,
        MasterProblemUserDescription(
            variables=variables, constraints=[assign, constraints], options=master_options
        ),
        [
            SubproblemUserDescription(
                k; variables=x[k, :], constraints=knapsack[k], options=options(k)
            ) for k in axes(x, 1)
        ],
    )
end

# The message of the error raised by `f`, `nothing` if it raises none.
function _error_of(f)
    try
        f()
        return nothing
    catch e
        return e
    end
end

function _build_gap_jump_model()
    cost = [
        5.0 8.0 14.0 20.0 5.0 4.0 13.0;
        18.0 14.0 15.0 16.0 3.0 8.0 19.0
    ]
    weight = [
        1.0 1.0 1.0 5.0 2.0 1.0 4.0;
        5.0 3.0 4.0 1.0 4.0 1.0 1.0
    ]
    capacity = [11.0, 14.0]
    K = 1:2
    T = 1:7

    model = Model()
    @variable(model, x[k in K, t in T], Bin)
    @constraint(model, assign[t in T], sum(x[k, t] for k in K) == 1)
    @constraint(
        model, knapsack[k in K], sum(weight[k, t] * x[k, t] for t in T) <= capacity[k]
    )
    @objective(model, Min, sum(cost[k, t] * x[k, t] for k in K, t in T))
    return model
end


function test_dml_basic_gap()
    @testset "[dml] dantzig_wolfe_decomposition basic GAP structure" begin
        model = _build_gap_jump_model()

        decomp, sp_map = _decompose_gap(model)

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

function test_dml_descriptions()
    @testset "[dml] a description takes any JuMP container" begin
        K, T = ["a", "b"], 3:5
        auto = JuMP.Containers.AutoContainerType
        for container in
            (auto, JuMP.Containers.DenseAxisArray, JuMP.Containers.SparseAxisArray, Dict)
            model = Model()
            @variable(model, x[k in K, t in T], container = container)
            @variable(model, s)
            @constraint(model, c[k in K, t in T], x[k, t] <= 1, container = container)
            @constraint(model, scalar_c, s <= 1)

            master = MasterProblemUserDescription(
                variables=[x, s], constraints=(c, [scalar_c])
            )
            @test Set(master.owned_vars) == Set([values(x)..., s])
            @test Set(master.owned_constrs) == Set([values(c)..., scalar_c])
        end

        model = _build_gap_jump_model()
        x = model[:x]
        slice = SubproblemUserDescription(1; variables=x[1, :])
        @test slice.id == 1
        @test slice.options == SubproblemOptions()
        @test Set(slice.owned_vars) == Set(x[1, t] for t in 1:7)
        @test isempty(slice.owned_constrs)
    end

    @testset "[dml] a variable is not a constraint" begin
        model = _build_gap_jump_model()
        err = _error_of(() -> MasterProblemUserDescription(variables=model[:assign]))
        @test err isa ArgumentError
        @test occursin("expected a variable", err.msg)

        err = _error_of(() -> SubproblemUserDescription(1; constraints=model[:x]))
        @test err isa ArgumentError
        @test occursin("expected a constraint", err.msg)

        err = _error_of(() -> SubproblemUserDescription(1; variables=:x))
        @test err isa ArgumentError
    end

    @testset "[dml] options have defaults" begin
        @test MasterProblemOptions() == MasterProblemOptions(silent=true)
        @test SubproblemOptions() == SubproblemOptions(silent=true, user_overridden=false)
    end
end

function test_dml_pure_master_variables()
    @testset "[dml] pure master variables in GAP with penalty" begin
        cost = [
            8.0 5.0 11.0 21.0 6.0 5.0 19.0;
            1.0 12.0 11.0 12.0 14.0 8.0 5.0
        ]
        weight = [
            2.0 3.0 3.0 1.0 2.0 1.0 1.0;
            5.0 1.0 1.0 3.0 1.0 5.0 4.0
        ]
        capacity = [5.0, 8.0]
        penalty = 10.0
        K = 1:2
        T = 1:7

        model = Model()
        @variable(model, x[k in K, t in T], Bin)
        @variable(model, u[t in T], Bin)
        @constraint(model, assign[t in T], sum(x[k, t] for k in K) + u[t] == 1)
        @constraint(
            model, knapsack[k in K], sum(weight[k, t] * x[k, t] for t in T) <= capacity[k]
        )
        @objective(
            model,
            Min,
            sum(cost[k, t] * x[k, t] for k in K, t in T) + sum(penalty * u[t] for t in T)
        )

        decomp, sp_map = _decompose_gap(model; variables=u)

        sp_ids = collect(subproblem_ids(decomp))
        @test length(sp_ids) == 2

        pm_vars = pure_master_variables(decomp)
        @test length(pm_vars) == 7

        @test all(pmv -> pure_master_cost(decomp, pmv) ≈ penalty, pm_vars)
        @test all(pm_vars) do pmv
            lb, ub = pure_master_bounds(decomp, pmv)
            lb ≈ 0.0 && ub ≈ 1.0
        end
        @test all(pmv -> Vertigo.Decomposition.pure_master_is_integer(decomp, pmv), pm_vars)
    end
end

function test_dml_silent()
    @testset "[dml] silent is chosen per master and subproblem" begin
        silent(m) = MOI.get(m, MOI.Silent())

        # Default: everything is silent.
        model = _build_gap_jump_model()
        set_optimizer(model, HiGHS.Optimizer)
        decomp, sp_map = _decompose_gap(model)
        @test silent(decomp.master_model)
        @test all(silent, values(sp_models(decomp)))

        # Explicit choice per model.
        decomp, sp_map = _decompose_gap(
            model;
            master_options=MasterProblemOptions(silent=false),
            options=k -> SubproblemOptions(silent=(k == 2)),
        )
        @test !silent(decomp.master_model)
        @test silent(sp_models(decomp)[sp_map[2]])
        @test !silent(sp_models(decomp)[sp_map[1]])
    end
end

function test_dml_user_overridden_subproblem()
    @testset "[dml] user-overridden subproblem has no model" begin
        model = _build_gap_jump_model()
        set_optimizer(model, HiGHS.Optimizer)
        decomp, sp_map = _decompose_gap(
            model; options=k -> SubproblemOptions(user_overridden=(k == 1))
        )

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

        # Every subproblem overridden.
        decomp, sp_map = _decompose_gap(
            model; options=k -> SubproblemOptions(user_overridden=true)
        )
        @test all(isnothing, values(sp_models(decomp)))
        @test length(coupling_constraints(decomp)) == 7

        # Column generation refuses to start until every pricer is installed.
        @test_throws "has no pricer" ColGenWorkspace(decomp, ColGenConfig())
    end
end

function test_dml_pure_master_bounds_and_types()
    @testset "[dml] pure master variables keep their bounds and integrality" begin
        model = Model()
        @variable(model, x[k in 1:2, t in 1:3], Bin)
        @variable(model, 2 <= u <= 7)
        @variable(model, 0 <= v <= 5, Int)
        @variable(model, w >= 0)
        @variable(model, b, Bin)
        @constraint(model, assign[t in 1:3], sum(x[k, t] for k in 1:2) + u + v + w + b >= 1)
        @constraint(model, knapsack[k in 1:2], sum(x[k, t] for t in 1:3) <= 2)
        # Distinct costs identify each master variable in the decomposition.
        @objective(model, Min, sum(x) + 1u + 2v + 3w + 4b)

        decomp, _ = _decompose_gap(model; variables=[u, v, w, b])

        by_cost = Dict(
            round(Int, pure_master_cost(decomp, pmv)) => pmv for
            pmv in pure_master_variables(decomp)
        )
        @test sort(collect(keys(by_cost))) == [1, 2, 3, 4]
        is_int(c) = Vertigo.Decomposition.pure_master_is_integer(decomp, by_cost[c])
        bounds(c) = pure_master_bounds(decomp, by_cost[c])

        @test bounds(1) == (2.0, 7.0) && !is_int(1)        # u: continuous
        @test bounds(2) == (0.0, 5.0) && is_int(2)         # v: integer
        @test bounds(3) == (0.0, Inf) && !is_int(3)        # w: no upper bound
        @test bounds(4) == (0.0, 1.0) && is_int(4)         # b: binary
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
        function decompose(model)
            return dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(constraints=model[:cover]),
                [SubproblemUserDescription(1; variables=model[:y])],
            )
        end

        decompose(build(10))   # compile everything but the size effect
        n = 2000
        model = build(n)
        elapsed = @elapsed decomp, _ = decompose(model)
        @test length(subproblem_variables(decomp, PricingSubproblemId(1))) == n
        @test elapsed < 10   # tens of seconds with the splat; about a second without
    end
end

# The decomposition must not change the problem: with every feasible column of every
# subproblem enumerated, the master (columns binary) has the optimum of the original model.
function _random_gap_instance(rng)
    K, T = rand(rng, 2:3), rand(rng, 4:6)
    return (
        K=K,
        T=T,
        cost=rand(rng, 1:20, K, T),
        weight=rand(rng, 1:5, K, T),
        capacity=rand(rng, 5:9, K),
    )
end

# `Ks` and `Ts` are the index sets the model is declared over, and `container` the JuMP
# container its objects are stored in; both must make no difference to the decomposition.
function _gap_model_for_decomposition(inst, penalty, Ks, Ts, container)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    kpos = Dict(k => i for (i, k) in enumerate(Ks))
    tpos = Dict(t => i for (i, t) in enumerate(Ts))
    cost(k, t) = inst.cost[kpos[k], tpos[t]]
    weight(k, t) = inst.weight[kpos[k], tpos[t]]
    @variable(model, x[k in Ks, t in Ts], Bin, container = container)
    @variable(model, u[t in Ts], Bin, container = container)
    @constraint(
        model,
        assign[t in Ts],
        sum(x[k, t] for k in Ks) + (penalty ? u[t] : 0) == 1,
        container = container
    )
    if !(penalty)
        @constraint(model, no_penalty[t in Ts], u[t] == 0, container = container)
    end
    @constraint(
        model,
        knapsack[k in Ks],
        sum(weight(k, t) * x[k, t] for t in Ts) <= inst.capacity[kpos[k]],
        container = container
    )
    @objective(
        model,
        Min,
        sum(cost(k, t) * x[k, t] for k in Ks, t in Ts) + sum(30 * u[t] for t in Ts)
    )
    return model
end

function _enumerated_master_optimum(inst, penalty, user_overridden, Ks, Ts, container)
    model = _gap_model_for_decomposition(inst, penalty, Ks, Ts, container)
    # Indexing by the declared keys works for every container.
    x, u, assign, knapsack = model[:x], model[:u], model[:assign], model[:knapsack]
    master_constrs = Any[[assign[t] for t in Ts]]
    if !penalty
        push!(master_constrs, [model[:no_penalty][t] for t in Ts])
    end
    decomp, sp_map = dantzig_wolfe_decomposition(
        model,
        MasterProblemUserDescription(
            variables=[u[t] for t in Ts], constraints=master_constrs
        ),
        [
            SubproblemUserDescription(
                k;
                variables=[x[k, t] for t in Ts],
                constraints=knapsack[k],
                options=SubproblemOptions(user_overridden=user_overridden),
            ) for k in Ks
        ],
    )
    @test Set(keys(sp_map)) == Set(Ks)   # subproblems are named by the declared indices
    master = master_model(decomp)
    for (ki, k) in enumerate(Ks)
        sp_id = sp_map[k]
        sp_vars = [last(mapped_subproblem_var(decomp, x[k, t])) for t in Ts]
        for mask in 0:(2 ^ inst.T - 1)
            positions = [ti for ti in 1:inst.T if (mask >> (ti - 1)) & 1 == 1]
            if !(sum(inst.weight[ki, ti] for ti in positions; init=0) <= inst.capacity[ki])
                continue
            end
            sol = Vertigo.Decomposition._SpSolution(
                sp_id, 0.0, [(sp_vars[ti], 1.0) for ti in positions]
            )
            coeffs = Dict{Vertigo.TaggedCI,Float64}(
                cid => c for
                (cid, c) in Vertigo.Decomposition.compute_column_coupling_coefficients(
                    decomp, sp_id, sol
                )
            )
            coeffs[convexity_ub_ci(decomp, sp_id)] = 1.0
            coeffs[convexity_lb_ci(decomp, sp_id)] = 1.0
            Vertigo.ColGen.add_variable!(
                master;
                variable_type=MOI.ZeroOne(),
                constraint_coeffs=coeffs,
                objective_coeff=Vertigo.Decomposition.compute_column_original_cost(
                    decomp, sp_id, sol
                ),
            )
        end
    end
    MOI.optimize!(master)
    status = MOI.get(master, MOI.TerminationStatus())
    return status == MOI.OPTIMAL ? MOI.get(master, MOI.ObjectiveValue()) : nothing, model
end

# The same instance declared over different index sets and containers.
function _gap_index_cases(inst)
    auto = JuMP.Containers.AutoContainerType
    dense, sparse = JuMP.Containers.DenseAxisArray, JuMP.Containers.SparseAxisArray
    names = ["m$k" for k in 1:inst.K]
    ints = (collect(1:inst.K), collect(1:inst.T))
    labelled = (names, collect(0:(inst.T - 1)))      # string machines, zero-based tasks
    shifted = (collect(1:inst.K), collect(5:(inst.T + 4)))
    return [
        ("integers, auto", ints..., auto),
        ("integers, Array", ints..., Array),
        ("strings and zero-based, auto", labelled..., auto),
        ("strings and zero-based, DenseAxisArray", labelled..., dense),
        ("strings and zero-based, SparseAxisArray", labelled..., sparse),
        ("strings and zero-based, Dict", labelled..., Dict),
        ("offset tasks, Dict", shifted..., Dict),
        ("integers, SparseAxisArray", ints..., sparse),
    ]
end

function test_dml_decomposition_matches_flat_model()
    @testset "[dml] master over enumerated columns has the original optimum" begin
        rng = Random.MersenneTwister(20261022)
        feasible = 0
        for _ in 1:3, penalty in (false, true), user_overridden in (false, true)
            inst = _random_gap_instance(rng)
            for (label, Ks, Ts, container) in _gap_index_cases(inst)
                @testset "$label" begin
                    master_obj, model = _enumerated_master_optimum(
                        inst, penalty, user_overridden, Ks, Ts, container
                    )
                    optimize!(model)
                    flat_obj = if termination_status(model) == MOI.OPTIMAL
                        objective_value(model)
                    else
                        nothing
                    end
                    @test isnothing(master_obj) == isnothing(flat_obj)
                    if !isnothing(flat_obj) && !isnothing(master_obj)
                        @test master_obj ≈ flat_obj atol = 1e-6
                        feasible += 1
                    end
                end
            end
        end
        # Not vacuous: the instances with a penalty variable are always feasible.
        @test feasible >= 12 * 8 / 2
    end
end

function test_dml_ownership_errors()
    @testset "[dml] a missing owner names the object" begin
        model = _build_gap_jump_model()
        x = model[:x]
        err = _error_of(
            () -> dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(constraints=model[:assign]),
                [SubproblemUserDescription(k; variables=x[k, :]) for k in 1:2],
            ),
        )
        @test err isa ArgumentError
        @test occursin("knapsack[1]", err.msg) || occursin("knapsack[2]", err.msg)
        @test occursin("constraint", err.msg)

        err = _error_of(
            () -> dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(constraints=model[:assign]),
                [
                    SubproblemUserDescription(k; constraints=model[:knapsack][k]) for
                    k in 1:2
                ],
            ),
        )
        @test err isa ArgumentError
        @test occursin("variable `x[", err.msg)
    end

    @testset "[dml] a variable or constraint owned twice is rejected" begin
        model = _build_gap_jump_model()
        x, knapsack = model[:x], model[:knapsack]
        subproblems(; extra=()) = [
            SubproblemUserDescription(
                k; variables=x[k, :], constraints=[knapsack[k], extra...]
            ) for k in 1:2
        ]

        # Owned by the master and by a subproblem.
        err = _error_of(
            () -> dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(constraints=[model[:assign], knapsack[1]]),
                subproblems(),
            ),
        )
        @test err isa ArgumentError
        @test occursin("`knapsack[1]` is assigned more than once", err.msg)
        @test occursin("the master", err.msg) && occursin("subproblem 1", err.msg)

        # Listed twice in one description.
        err = _error_of(
            () -> dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(constraints=model[:assign]),
                subproblems(; extra=[knapsack[1]]),
            ),
        )
        @test occursin("`knapsack[1]` is assigned more than once", err.msg)
    end

    @testset "[dml] subproblems need distinct identifiers" begin
        model = _build_gap_jump_model()
        x, knapsack = model[:x], model[:knapsack]
        err = _error_of(
            () -> dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(constraints=model[:assign]),
                [
                    SubproblemUserDescription(1; variables=x[1, :], constraints=knapsack[1]),
                    SubproblemUserDescription(1; variables=x[2, :], constraints=knapsack[2]),
                ],
            ),
        )
        @test err isa ArgumentError
        @test occursin("subproblem 1 is described more than once", err.msg)
    end

    @testset "[dml] no subproblem is rejected" begin
        model = _build_gap_jump_model()
        err = _error_of(
            () -> dantzig_wolfe_decomposition(
                model, MasterProblemUserDescription(constraints=model[:assign]), []
            ),
        )
        @test err isa ArgumentError
        @test occursin("At least one subproblem", err.msg)
    end

    @testset "[dml] a reference of another model is rejected" begin
        model = _build_gap_jump_model()
        other = _build_gap_jump_model()
        err = _error_of(
            () -> dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(constraints=model[:assign]),
                [
                    SubproblemUserDescription(
                        k; variables=other[:x][k, :], constraints=model[:knapsack][k]
                    ) for k in 1:2
                ],
            ),
        )
        @test err isa ArgumentError
        @test occursin("does not belong to the model", err.msg)
    end

    @testset "[dml] unnamed variables and constraints are assigned by reference" begin
        model = _build_gap_jump_model()
        extra = @variable(model, [1:2], lower_bound = 0)
        extra_constr = @constraint(model, sum(model[:x][1, t] for t in 1:7) <= 3)

        err = _error_of(() -> _decompose_gap(model))
        @test err isa ArgumentError
        @test occursin("variable", err.msg)

        err = _error_of(() -> _decompose_gap(model; variables=extra))
        @test err isa ArgumentError
        @test occursin("constraint", err.msg)

        decomp, _ = _decompose_gap(model; variables=extra, constraints=extra_constr)
        @test length(pure_master_variables(decomp)) == 2
    end

    @testset "[dml] every JuMP container is decomposed by its real indices" begin
        # x[k, t]: two subproblems "a" and "b" over tasks 3:5, linked by cover[t].
        # The decomposition must be the same whatever container JuMP stores them in.
        for container in
            (JuMP.Containers.DenseAxisArray, JuMP.Containers.SparseAxisArray, Dict)
            model = Model()
            K, T = ["a", "b"], 3:5
            x = @variable(model, [k in K, t in T], base_name = "x", container = container)
            cover = @constraint(
                model, [t in T], sum(x[k, t] for k in K) == 1, container = container
            )
            cap = @constraint(
                model, [k in K], sum(x[k, t] for t in T) <= 2, container = Dict
            )
            @objective(model, Min, sum(x[k, t] for k in K, t in T))
            decomp, sp_map = dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(constraints=cover),
                [
                    SubproblemUserDescription(
                        k; variables=[x[k, t] for t in T], constraints=cap[k]
                    ) for k in K
                ],
            )
            @test sort(collect(keys(sp_map))) == ["a", "b"]
            for id in subproblem_ids(decomp)
                @test length(subproblem_variables(decomp, id)) == 3
            end
            @test length(coupling_constraints(decomp)) == 3
        end
    end

    @testset "[dml] an identifier can be any sortable value" begin
        model = Model()
        @variable(model, y[m in 1:2, t in 1:3], Bin)
        @constraint(model, link[t in 1:3], sum(y[m, t] for m in 1:2) == 1)
        @constraint(model, cap[m in 1:2, t in 1:3], y[m, t] <= 1)
        @objective(model, Min, sum(y))
        decomp, sp_map = dantzig_wolfe_decomposition(
            model,
            MasterProblemUserDescription(constraints=link),
            [
                SubproblemUserDescription(
                    (m, t); variables=y[m, t], constraints=cap[m, t]
                ) for m in 1:2 for t in 1:3
            ],
        )
        @test sort(collect(keys(sp_map))) == [(m, t) for m in 1:2 for t in 1:3]
        @test all(length(subproblem_variables(decomp, id)) == 1 for id in values(sp_map))
    end

    @testset "[dml] subproblem constraint on a foreign variable is rejected" begin
        model = _build_gap_jump_model()
        x, knapsack = model[:x], model[:knapsack]
        # knapsack[1] is declared on subproblem 1 but also uses x[2, 1].
        mixed = @constraint(model, mixed, x[1, 1] + x[2, 1] <= 1)
        err = _error_of(
            () -> dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(constraints=model[:assign]),
                [
                    SubproblemUserDescription(
                        1; variables=x[1, :], constraints=[knapsack[1], mixed]
                    ),
                    SubproblemUserDescription(
                        2; variables=x[2, :], constraints=knapsack[2]
                    ),
                ],
            ),
        )
        @test err isa ArgumentError
        @test occursin("`mixed`", err.msg)
        @test occursin("subproblem 2", err.msg)

        # A master variable in a subproblem constraint is just as foreign.
        model = _build_gap_jump_model()
        x, knapsack = model[:x], model[:knapsack]
        z = @variable(model, z >= 0)
        withz = @constraint(model, withz, x[1, 1] + z <= 1)
        err = _error_of(
            () -> dantzig_wolfe_decomposition(
                model,
                MasterProblemUserDescription(variables=z, constraints=model[:assign]),
                [
                    SubproblemUserDescription(
                        1; variables=x[1, :], constraints=[knapsack[1], withz]
                    ),
                    SubproblemUserDescription(
                        2; variables=x[2, :], constraints=knapsack[2]
                    ),
                ],
            ),
        )
        @test err isa ArgumentError
        @test occursin("the master", err.msg)
    end
end

function test_dml()
    test_dml_basic_gap()
    test_dml_descriptions()
    test_dml_pure_master_variables()
    test_dml_silent()
    test_dml_user_overridden_subproblem()
    test_dml_pure_master_bounds_and_types()
    test_dml_large_subproblem_objective()
    test_dml_decomposition_matches_flat_model()
    return test_dml_ownership_errors()
end
