# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module ColumnGenerationUnitTests

using Test
using Random
using JuMP
using HiGHS
using MathOptInterface
const MOI = MathOptInterface

using ColumnGeneration
using ColumnGeneration: MasterPrimalSolution, PrimalMoiSolution,
    check_primal_ip_feasibility!, update_inc_primal_sol!, Phase0, Phase1, Phase2

# ────────────────────────────────────────────────────────────────────────────────────────
# GAP Instance helpers
# ────────────────────────────────────────────────────────────────────────────────────────

struct GAPInstance
    n_machines::Int
    n_tasks::Int
    cost::Matrix{Float64}      # cost[k,t]
    weight::Matrix{Float64}    # weight[k,t]
    capacity::Vector{Float64}  # capacity[k]
end

function random_gap_instance(n_machines, n_tasks; seed=42)
    rng = Random.MersenneTwister(seed)
    cost = rand(rng, 1:20, n_machines, n_tasks) .* 1.0
    weight = rand(rng, 1:5, n_machines, n_tasks) .* 1.0
    capacity = [ceil(0.7 * sum(weight[k, :])) for k in 1:n_machines]
    return GAPInstance(n_machines, n_tasks, cost, weight, capacity)
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Build ColGenContext for a GAP instance
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    build_gap_context(inst) -> ColGenContext

Build a column generation context for the given GAP instance.

The master model contains:
  - Assignment coupling constraints: Σₖ xₖₜ = 1 for each task t (EqualTo)
  - Convexity constraints: Σλ ≥ 0 (GreaterThan) and Σλ ≤ 1 (LessThan) per machine

Each subproblem is a binary knapsack for one machine.
"""
function build_gap_context(inst::GAPInstance)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    # ── Master model ──────────────────────────────────────────────────────────
    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)

    @constraint(master_jump, assignment[t in T], 0 == 1)   # coupling: = 1
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)      # Σλ ≥ 0
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)      # Σλ ≤ 1
    @objective(master_jump, Min, 0)

    master_model = backend(master_jump)

    # ── Subproblem models ─────────────────────────────────────────────────────
    sp_models = Dict{Any,Any}()
    sp_var_indices = Dict{Int,Vector{MOI.VariableIndex}}()

    for k in K
        sp_jump = Model(HiGHS.Optimizer)
        set_silent(sp_jump)

        @variable(sp_jump, z[t in T], Bin)
        @constraint(sp_jump, sum(inst.weight[k, t] * z[t] for t in T) <= inst.capacity[k])
        @objective(sp_jump, Min, sum(inst.cost[k, t] * z[t] for t in T))

        sp_models[k] = backend(sp_jump)
        sp_var_indices[k] = [index(z[t]) for t in T]
    end

    # ── Build Decomposition ───────────────────────────────────────────────────
    SpVar = MOI.VariableIndex
    CstrId = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}

    builder = DecompositionBuilder{Int,SpVar,Tuple{Int,Int},CstrId,Nothing}(minimize=true)

    for k in K
        add_subproblem!(builder, k, 0.0, 0.0, 1.0)
    end

    for k in K
        for t in T
            sp_var = sp_var_indices[k][t]
            add_sp_variable!(builder, k, sp_var, inst.cost[k, t])
            cstr_idx = index(assignment[t])
            add_coupling_coefficient!(builder, k, sp_var, cstr_idx, 1.0)
            add_mapping!(builder, (k, t), k, sp_var)
        end
    end

    for t in T
        add_coupling_constraint!(builder, index(assignment[t]), EQUAL_TO, 1.0)
    end

    decomp = build(builder)

    # ── Column pool ───────────────────────────────────────────────────────────
    pool = ColumnPool{MOI.VariableIndex,Int,SpVar}()

    # ── Convexity constraint indices ──────────────────────────────────────────
    conv_ub_map = Dict{Any,Any}(k => index(conv_ub[k]) for k in K)
    conv_lb_map = Dict{Any,Any}(k => index(conv_lb[k]) for k in K)

    # ── Build context ─────────────────────────────────────────────────────────
    ctx = ColGenContext(
        decomp,
        master_model,
        conv_ub_map,
        conv_lb_map,
        sp_models,
        pool,
        NonRobustCutManager{CstrId}(),
        Dict{Any,Any}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )

    return ctx
end

include("ip_management_tests.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────────────────

function test_gap_decomposition_builder()
    @testset "[gap] decomposition builder" begin
        inst = random_gap_instance(2, 4)
        ctx = build_gap_context(inst)

        @test length(collect(subproblem_ids(ctx.decomp))) == 2
        for k in 1:2
            vars = subproblem_variables(ctx.decomp, k)
            @test length(vars) == 4
            lb, ub = convexity_bounds(ctx.decomp, k)
            @test lb ≈ 0.0
            @test ub ≈ 1.0
            @test subproblem_fixed_cost(ctx.decomp, k) ≈ 0.0
        end
        @test length(coupling_constraints(ctx.decomp)) == 4
        @test is_minimization(ctx.decomp)
    end
end

function test_gap_column_pool_populated()
    @testset "[gap] column pool is populated after CG" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst)

        run_column_generation(ctx)

        # Pool must have columns — at least one per machine
        @test length(ctx.pool.by_master_var) >= 2
    end
end

function test_gap_lp_dual_bound_matches_primal()
    @testset "[gap] LP dual bound approximately equals primal at convergence" begin
        inst = random_gap_instance(2, 7)
        ctx = build_gap_context(inst)

        output = run_column_generation(ctx)

        @test !isnothing(output.incumbent_dual_bound)
        @test !isnothing(output.master_lp_obj)

        gap = abs(output.master_lp_obj - output.incumbent_dual_bound)
        @test gap <= 1.0  # within 1 unit (tight for LP relaxation)
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function run()
    test_gap_decomposition_builder()
    test_gap_column_pool_populated()
    test_gap_lp_dual_bound_matches_primal()
    test_ip_management()
end

end # module ColumnGenerationUnitTests
