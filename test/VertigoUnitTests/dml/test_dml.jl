# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# Ownerships written by hand, as `@dantzig_wolfe` would build them.
function _master_of(name, nindices)
    return Vertigo.DML.Ownership(
        name,
        nindices,
        true,
        (idx...) -> (id=nothing, silent=nothing, user_overriden=nothing),
    )
end
function _subproblem_of(name, nindices, id)
    return Vertigo.DML.Ownership(
        name,
        nindices,
        false,
        (idx...) -> (id=id(idx...), silent=nothing, user_overriden=nothing),
    )
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

function test_dml_macro_basic_gap()
    @testset "[dml] @dantzig_wolfe basic GAP structure" begin
        model = _build_gap_jump_model()

        decomp, sp_map = @dantzig_wolfe model begin
            subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
            master() => (constraints = assign[_])
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
    @testset "[dml] dantzig_wolfe_decomposition with ownerships" begin
        model = _build_gap_jump_model()

        ownerships = [
            _subproblem_of(:x, 2, (k, _) -> k),
            _master_of(:assign, 1),
            _subproblem_of(:knapsack, 1, k -> k),
        ]

        decomp, sp_map = dantzig_wolfe_decomposition(model, ownerships)

        sp_ids = collect(subproblem_ids(decomp))
        @test length(sp_ids) == 2

        @test length(subproblem_variables(decomp, sp_ids[1])) == 7
        @test length(subproblem_variables(decomp, sp_ids[2])) == 7

        @test length(coupling_constraints(decomp)) == 7
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

        decomp, sp_map = @dantzig_wolfe model begin
            subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
            master() => (variables=u[_], constraints=assign[_])
        end

        sp_ids = collect(subproblem_ids(decomp))
        @test length(sp_ids) == 2

        pm_vars = pure_master_variables(decomp)
        @test length(pm_vars) == 7

        @test all(pmv -> pure_master_cost(decomp, pmv) ≈ penalty, pm_vars)
        @test all(pm_vars) do pmv
            lb, ub = pure_master_bounds(decomp, pmv)
            lb ≈ 0.0 && ub ≈ 1.0
        end
        @test all(pmv -> Vertigo.Reformulation.pure_master_is_integer(decomp, pmv), pm_vars)
    end
end

function test_dml_silent()
    @testset "[dml] silent is chosen per master and subproblem" begin
        silent(m) = MOI.get(m, MOI.Silent())

        # Default: everything is silent.
        model = _build_gap_jump_model()
        set_optimizer(model, HiGHS.Optimizer)
        decomp, sp_map = @dantzig_wolfe model begin
            subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
            master() => (constraints = assign[_])
        end
        @test silent(decomp.master_model)
        @test all(silent, values(sp_models(decomp)))

        # Explicit choice per model; one rule is enough to decide.
        decomp, sp_map = @dantzig_wolfe model begin
            subproblem(k, silent=(k == 2)) => (variables = x[k, _])
            master(silent=false) => (constraints = assign[_])
            subproblem(k) => (constraints = knapsack[k])
        end
        @test !silent(decomp.master_model)
        @test silent(sp_models(decomp)[sp_map[2]])
        @test !silent(sp_models(decomp)[sp_map[1]])

        # Rules assigning to the same model must agree.
        @test_throws ErrorException @dantzig_wolfe model begin
            subproblem(k, silent=true) => (variables = x[k, _])
            master() => (constraints = assign[_])
            subproblem(k, silent=false) => (constraints = knapsack[k])
        end
    end
end

function test_dml_user_overriden_subproblem()
    @testset "[dml] user-overridden subproblem has no model" begin
        model = _build_gap_jump_model()
        set_optimizer(model, HiGHS.Optimizer)
        decomp, sp_map = @dantzig_wolfe model begin
            subproblem(k, user_overriden=(k == 1)) => (variables = x[k, _])
            master() => (constraints = assign[_])
            subproblem(k) => (constraints = knapsack[k])
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
            subproblem(k, user_overriden=true) =>
                (variables=[x[k, _]], constraints=knapsack[k])
        end
        @test all(isnothing, values(sp_models(decomp)))
        @test length(coupling_constraints(decomp)) == 7

        # Column generation refuses to start until every pricer is installed.
        @test_throws "has no pricer" ColGenWorkspace(decomp, ColGenConfig())

        # Rules assigning to the same subproblem must agree.
        @test_throws ErrorException @dantzig_wolfe model begin
            subproblem(k, user_overriden=true) => (variables = x[k, _])
            master() => (constraints = assign[_])
            subproblem(k, user_overriden=false) => (constraints = knapsack[k])
        end
    end
end

function test_dml_unknown_option()
    @testset "[dml] an option the model does not take is rejected" begin
        err = try
            @macroexpand @dantzig_wolfe Model() begin
                master(user_overriden=true) => (constraints = assign[_])
            end
            nothing
        catch e
            e
        end
        @test occursin("Unknown option `user_overriden` for master", sprint(showerror, err))
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

        decomp, _ = @dantzig_wolfe model begin
            subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
            master() => (variables=[u, v, w, b], constraints=assign[_])
        end

        by_cost = Dict(
            round(Int, pure_master_cost(decomp, pmv)) => pmv for
            pmv in pure_master_variables(decomp)
        )
        @test sort(collect(keys(by_cost))) == [1, 2, 3, 4]
        is_int(c) = Vertigo.Reformulation.pure_master_is_integer(decomp, by_cost[c])
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
        ownerships = [_subproblem_of(:y, 1, i -> 1), _master_of(:cover, 0)]

        dantzig_wolfe_decomposition(build(10), ownerships)   # compile everything but the size effect
        n = 2000
        elapsed = @elapsed decomp, _ = dantzig_wolfe_decomposition(build(n), ownerships)
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

function _enumerated_master_optimum(inst, penalty, user_overriden, Ks, Ts, container)
    model = _gap_model_for_decomposition(inst, penalty, Ks, Ts, container)
    # Option values are evaluated inside Vertigo.DML, so they cannot name local variables.
    decomp, sp_map = if user_overriden
        @dantzig_wolfe model begin
            subproblem(k, user_overriden=true) => (variables = x[k, _])
            master() => (variables=u[_], constraints=[assign[_], no_penalty[_]])
            subproblem(k) => (constraints = knapsack[k])
        end
    else
        @dantzig_wolfe model begin
            subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
            master() => (variables=u[_], constraints=[assign[_], no_penalty[_]])
        end
    end
    @test Set(keys(sp_map)) == Set(Ks)   # subproblems are named by the declared indices
    master = master_model(decomp)
    x = model[:x]
    for (ki, k) in enumerate(Ks)
        sp_id = sp_map[k]
        sp_vars = [last(mapped_subproblem_var(decomp, x[k, t])) for t in Ts]
        for mask in 0:(2 ^ inst.T - 1)
            positions = [ti for ti in 1:inst.T if (mask >> (ti - 1)) & 1 == 1]
            if !(sum(inst.weight[ki, ti] for ti in positions; init=0) <= inst.capacity[ki])
                continue
            end
            sol = Vertigo.Reformulation._SpSolution(
                sp_id, 0.0, [(sp_vars[ti], 1.0) for ti in positions]
            )
            coeffs = Dict{Vertigo.TaggedCI,Float64}(
                cid => c for
                (cid, c) in Vertigo.Reformulation.compute_column_coupling_coefficients(
                    decomp, sp_id, sol
                )
            )
            coeffs[convexity_ub_ci(decomp, sp_id)] = 1.0
            coeffs[convexity_lb_ci(decomp, sp_id)] = 1.0
            Vertigo.ColGen.add_variable!(
                master;
                variable_type=MOI.ZeroOne(),
                constraint_coeffs=coeffs,
                objective_coeff=Vertigo.Reformulation.compute_column_original_cost(
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
        for _ in 1:3, penalty in (false, true), user_overriden in (false, true)
            inst = _random_gap_instance(rng)
            for (label, Ks, Ts, container) in _gap_index_cases(inst)
                @testset "$label" begin
                    master_obj, model = _enumerated_master_optimum(
                        inst, penalty, user_overriden, Ks, Ts, container
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

# The contract the macro and annotation functions rely on: a rule is called for every element
# of every variable and constraint, with the indices as they were declared (it may be called
# more than once per element, so rules must be pure).
function test_dml_rules_see_declared_indices()
    @testset "[dml] rules receive the declared indices of every container" begin
        K, T = ["a", "b"], 3:5
        declared = Set((k, t) for k in K for t in T)
        auto = JuMP.Containers.AutoContainerType
        for container in
            (auto, JuMP.Containers.DenseAxisArray, JuMP.Containers.SparseAxisArray, Dict)
            model = Model()
            @variable(model, x[k in K, t in T], container = container)
            @constraint(model, c[k in K, t in T], x[k, t] <= 1, container = container)
            @variable(model, s)                       # scalar
            @constraint(model, scalar_c, s <= 1)
            @objective(model, Min, sum(x[k, t] for k in K, t in T) + s)

            seen = Dict{Symbol,Vector{Any}}()
            record(name, idx...) = push!(get!(Vector{Any}, seen, name), idx)
            ownerships = [
                _subproblem_of(:x, 2, (k, t) -> (record(:x, k, t); k)),
                _subproblem_of(:c, 2, (k, t) -> (record(:c, k, t); k)),
                _subproblem_of(:s, 0, () -> (record(:s); "a")),
                _subproblem_of(:scalar_c, 0, () -> (record(:scalar_c); "a")),
            ]
            dantzig_wolfe_decomposition(model, ownerships)

            for name in (:x, :c)
                @test Set(Tuple.(seen[name])) == declared
            end
            @test unique(seen[:s]) == [()] && unique(seen[:scalar_c]) == [()]
        end
    end
end

function test_dml_rule_errors()
    @testset "[dml] missing rule names the object" begin
        model = _build_gap_jump_model()
        err = try
            @dantzig_wolfe model begin
                subproblem(k) => (variables = x[k, _])
                master() => (constraints = assign[_])
            end
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("knapsack[1]", err.msg) || occursin("knapsack[2]", err.msg)
        @test occursin("constraint", err.msg)

        model = _build_gap_jump_model()
        err = try
            @dantzig_wolfe model begin
                master() => (constraints = assign[_])
                subproblem(k) => (constraints = knapsack[k])
            end
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("variable `x[", err.msg)
    end

    @testset "[dml] two rules on one name are rejected" begin
        err = try
            Core.eval(
                @__MODULE__,
                quote
                    @dantzig_wolfe _build_gap_jump_model() begin
                        subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
                        subproblem(t) => (variables = x[_, t])
                        master() => (constraints = assign[_])
                    end
                end,
            )
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("more than one rule", sprint(showerror, err))
    end

    @testset "[dml] parse_block" begin
        parse = Vertigo.DML.parse_block

        # A block with no `master()` line is rejected.
        err = try
            parse(:(
                begin
                    subproblem(k) => (variables = x[k, _])
                end
            ))
        catch e
            e
        end
        @test occursin("exactly one master()", sprint(showerror, err))

        # `master()` written twice.
        err = try
            parse(:(
                begin
                    master() => (constraints = assign[_])
                    master() => (variables = y)
                end
            ))
        catch e
            e
        end
        @test occursin("exactly one master()", sprint(showerror, err))
    end

    @testset "[dml] no ownership is rejected" begin
        err = try
            dantzig_wolfe_decomposition(_build_gap_jump_model(), [])
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("No slice descriptions specified", err.msg)
    end

    @testset "[dml] a master and a subproblem rule on one name are rejected" begin
        err = try
            Core.eval(@__MODULE__, quote
                @dantzig_wolfe _build_gap_jump_model() begin
                    master() => (constraints = assign[_])
                    subproblem(k) => (constraints = assign[k])
                end
            end)
        catch e
            e
        end
        @test occursin(
            "`assign` with 1 index(es) is matched by more than one rule",
            sprint(showerror, err),
        )
    end

    @testset "[dml] unnamed variables and constraints are rejected" begin
        model = _build_gap_jump_model()
        @variable(model, [1:2], lower_bound = 0)
        err = try
            @dantzig_wolfe model begin
                subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
                master() => (constraints = assign[_])
            end
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("variable", err.msg) && occursin("no name", err.msg)

        model = _build_gap_jump_model()
        @constraint(model, sum(model[:x][1, t] for t in 1:7) <= 3)
        err = try
            @dantzig_wolfe model begin
                subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
                master() => (constraints = assign[_])
            end
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("constraint", err.msg) && occursin("no name", err.msg)
    end

    @testset "[dml] every JuMP container is decomposed by its real indices" begin
        # x[k, t]: two subproblems "a" and "b" over tasks 3:5, linked by cover[t].
        # The decomposition must be the same whatever container JuMP stores them in.
        for container in
            (JuMP.Containers.DenseAxisArray, JuMP.Containers.SparseAxisArray, Dict)
            model = Model()
            K, T = ["a", "b"], 3:5
            x = @variable(model, [k in K, t in T], base_name = "x", container = container)
            model[:x] = x
            model[:cover] = @constraint(
                model, [t in T], sum(x[k, t] for k in K) == 1, container = container
            )
            @constraint(model, cap[k in K], sum(x[k, t] for t in T) <= 2, container = Dict)
            @objective(model, Min, sum(x[k, t] for k in K, t in T))
            decomp, sp_map = @dantzig_wolfe model begin
                subproblem(k) => (variables=x[k, _], constraints=cap[k])
                master() => (constraints = cover[_])
            end
            @test sort(collect(keys(sp_map))) == ["a", "b"]
            for id in subproblem_ids(decomp)
                @test length(subproblem_variables(decomp, id)) == 3
            end
            @test length(coupling_constraints(decomp)) == 3
        end
    end

    @testset "[dml] subproblem constraint on a foreign variable is rejected" begin
        model = _build_gap_jump_model()
        # knapsack[1] is declared on subproblem 1 but also uses x[2, 1].
        @constraint(model, mixed, model[:x][1, 1] + model[:x][2, 1] <= 1)
        err = try
            @dantzig_wolfe model begin
                subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
                master() => (constraints = assign[_])
                subproblem(1) => (constraints = mixed)
            end
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("`mixed`", err.msg)
        @test occursin("subproblem 2", err.msg)

        # A master variable in a subproblem constraint is just as foreign.
        model = _build_gap_jump_model()
        @variable(model, z >= 0)
        @constraint(model, withz, model[:x][1, 1] + z <= 1)
        err = try
            @dantzig_wolfe model begin
                subproblem(k) => (variables=x[k, _], constraints=knapsack[k])
                master() => (variables=z, constraints=assign[_])
                subproblem(1) => (constraints = withz)
            end
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("the master", err.msg)
    end
end

function test_dml()
    test_dml_user_overriden_subproblem()
    test_dml_macro_basic_gap()
    test_dml_silent()
    test_dml_annotation_function()
    test_dml_pure_master_variables()
    test_dml_unknown_option()
    test_dml_pure_master_bounds_and_types()
    test_dml_large_subproblem_objective()
    test_dml_decomposition_matches_flat_model()
    test_dml_rule_errors()
    return test_dml_rules_see_declared_indices()
end
