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

# Brute-force knapsack pricing oracle for one GAP machine: it stands in for the
# MOI model of a `user_overriden=true` subproblem, receiving the reduced costs through
# `MOI.modify` like a model would.
mutable struct KnapsackPricer
    vars::Vector{MOI.VariableIndex}   # vars[t]: subproblem variable of task t
    weight::Vector{Float64}
    capacity::Float64
    red_cost::Vector{Float64}
end

function MOI.modify(
    p::KnapsackPricer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    change::MOI.ScalarCoefficientChange{Float64}
)
    p.red_cost[findfirst(==(change.variable), p.vars)] = change.new_coefficient
    return nothing
end

function Vertigo.ColGen.get_pricing_subprob_optimizer(
    ::Vertigo.ColGen.ExactStage,
    sp::Vertigo.ColGen.PricingSubproblem{<:KnapsackPricer}
)
    return Vertigo.ColGen.moi_pricing_sp(sp)
end

function Vertigo.ColGen.optimize_pricing_problem!(
    ws::Vertigo.ColGen.ColGenWorkspace, sp_id,
    ::Vertigo.ColGen.PricingSubproblem, p::KnapsackPricer,
    mast_dual_sol, _stab_changes_mast_dual_sol
)
    CG = Vertigo.ColGen
    n = length(p.vars)
    best_obj, best_set = Inf, Int[]
    for mask in 0:(2^n - 1)
        items = [t for t in 1:n if (mask >> (t - 1)) & 1 == 1]
        sum(p.weight[items]) <= p.capacity || continue
        obj = sum(p.red_cost[items])
        obj < best_obj && ((best_obj, best_set) = (obj, items))
    end

    decomp = ws.decomp
    ν = CG._get_convexity_dual(mast_dual_sol, convexity_lb_pairs(decomp), sp_id) +
        CG._get_convexity_dual(mast_dual_sol, convexity_ub_pairs(decomp), sp_id)
    reduced_cost = best_obj + subproblem_fixed_cost(decomp, sp_id) - ν
    sol = Vertigo.Reformulation._SpSolution(
        sp_id, reduced_cost, [(p.vars[t], 1.0) for t in best_set]
    )
    improving = reduced_cost < -CG.RC_IMPROVING_TOL
    return CG.PricingSolution(
        false, false, reduced_cost, best_obj,
        [CG.PricingPrimalSolution(sp_id, sol, improving)]
    )
end

function test_dml_user_overriden_subproblem()
    @testset "[dml] GAP with user-overridden knapsack pricers dual ≈ 63.0" begin
        cost = [5.0 8.0 14.0 20.0 5.0 4.0 13.0;
                18.0 14.0 15.0 16.0 3.0 8.0 19.0]
        weight = [1.0 1.0 1.0 5.0 2.0 1.0 4.0;
                  5.0 3.0 4.0 1.0 4.0 1.0 1.0]
        capacity = [11.0, 14.0]
        K = 1:2; T = 1:7

        model = Model(HiGHS.Optimizer)
        @variable(model, x[k in K, t in T], Bin)
        @constraint(model, assign[t in T], sum(x[k, t] for k in K) == 1)
        @constraint(model, knapsack[k in K],
            sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
        @objective(model, Min, sum(cost[k, t] * x[k, t] for k in K, t in T))

        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k, user_overriden=true)
            knapsack[k] => subproblem(k)
            assign[_] => master()
        end

        # No model was built; the user installs a pricer per subproblem.
        for k in K
            sp_id = sp_map[k]
            @test sp_model(decomp, sp_id) === nothing
            vars = [last(mapped_subproblem_var(decomp, x[k, t])) for t in T]
            sp_models(decomp)[sp_id] = KnapsackPricer(
                vars, weight[k, :], capacity[k], zeros(length(T))
            )
        end

        ws = _build_colgen_context_from_dml(decomp, sp_map)
        output = run_column_generation(ws)
        @test output.status == optimal
        @test abs(output.incumbent_dual_bound - 63.0) <= 1e-4
    end
end

# ── Branch-and-price over every JuMP container ──────────────────────────────────────────
#
# The same GAP declared over different index sets and containers must decompose into the same
# problem: branch-and-price finds the optimum of the flat model, and every declaration has
# the root bound of the plain `1:n` one. Each instance has a fractional root (root bound
# below the optimum), so branching is exercised.

# (seed, machines, tasks, capacity)
const CONTAINER_GAP_INSTANCES = [(2, 3, 10, 9.0), (7, 3, 10, 9.0), (9, 4, 12, 9.0)]

function _container_gap_instance(seed, K, T, capacity)
    rng = Random.MersenneTwister(seed)
    return (
        K=K, T=T,
        cost=rand(rng, 1:20, K, T) .* 1.0, weight=rand(rng, 1:5, K, T) .* 1.0,
        capacity=fill(capacity, K),
    )
end

function test_dml_bp_over_containers()
    auto = JuMP.Containers.AutoContainerType
    dense, sparse = JuMP.Containers.DenseAxisArray, JuMP.Containers.SparseAxisArray
    for (seed, K, T, capacity) in CONTAINER_GAP_INSTANCES
        inst = _container_gap_instance(seed, K, T, capacity)
        ints = (collect(1:K), collect(1:T))
        labelled = (["m$k" for k in 1:K], collect(0:(T - 1)))   # string machines, zero-based tasks
        cases = [
            ("1:n Array", ints..., Array),
            ("1:n auto", ints..., auto),
            ("DenseAxisArray", labelled..., dense),
            ("SparseAxisArray", labelled..., sparse),
            ("Dict", labelled..., Dict),
        ]
        reference_root_lp = Ref{Union{Nothing,Float64}}(nothing)
        for (label, Ks, Ts, container) in cases
            @testset "[dml][bp] GAP seed $seed ($(K)x$(T)), $label" begin
                kpos = Dict(k => i for (i, k) in enumerate(Ks))
                tpos = Dict(t => j for (j, t) in enumerate(Ts))
                model = Model(HiGHS.Optimizer)
                set_silent(model)
                @variable(model, x[k in Ks, t in Ts], Bin, container = container)
                @constraint(
                    model, assign[t in Ts], sum(x[k, t] for k in Ks) == 1, container = container
                )
                @constraint(
                    model, knapsack[k in Ks],
                    sum(inst.weight[kpos[k], tpos[t]] * x[k, t] for t in Ts) <=
                    inst.capacity[kpos[k]],
                    container = container
                )
                @objective(
                    model, Min, sum(inst.cost[kpos[k], tpos[t]] * x[k, t] for k in Ks, t in Ts)
                )

                optimize!(model)
                @test termination_status(model) == MOI.OPTIMAL
                flat_optimum = objective_value(model)

                decomp, sp_map = @dantzig_wolfe model begin
                    x[k, _] => subproblem(k)
                    assign[_] => master()
                    knapsack[k] => subproblem(k)
                end
                @test Set(keys(sp_map)) == Set(Ks)

                bcp_ws = BranchCutPriceWorkspace(decomp, BranchCutPriceConfig(node_limit=500))
                output = redirect_stdout(devnull) do
                    run_branch_and_price(bcp_ws)
                end

                @test !isnothing(output.incumbent)
                if !isnothing(output.incumbent)
                    @test output.incumbent.obj_value ≈ flat_optimum atol = 1e-6
                end
                @test output.root_lp_value < flat_optimum - 1e-6   # the root is fractional...
                @test 1 < output.nodes_explored < 500              # ...so the search branched and finished
                # Known solver limitation, unrelated to containers: once the tree is exhausted the
                # global dual bound is -Inf and the status says :node_limit. Flips when fixed.
                @test_broken output.status == :optimal

                if isnothing(reference_root_lp[])
                    reference_root_lp[] = output.root_lp_value
                else
                    @test output.root_lp_value ≈ reference_root_lp[] atol = 1e-6
                end
            end
        end
    end
end

function test_dml_e2e()
    test_dml_bp_over_containers()
    test_dml_user_overriden_subproblem()
    for smoothing_alpha in (0.0, 0.5)
        test_dml_gap_converges(; smoothing_alpha)
        test_dml_gap_converges2(; smoothing_alpha)
        test_dml_maximization(; smoothing_alpha)
        test_dml_with_penalty(; smoothing_alpha)
        test_dml_with_penalty_cardinality(; smoothing_alpha)
        test_dml_fixed_master_cost(; smoothing_alpha)
        test_dml_dual_bound_pure_master(; smoothing_alpha)
    end
end
