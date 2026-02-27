# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module ColumnGenerationE2eTests

using Test
using Random
using JuMP
using HiGHS
using MathOptInterface
const MOI = MathOptInterface

using ColumnGeneration

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

# ────────────────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────────────────

function test_gap_column_generation_converges()
    @testset "[gap] column generation converges (2 machines, 7 tasks)" begin
        inst = random_gap_instance(2, 7)
        ctx = build_gap_context(inst)

        output = run_column_generation(ctx)

        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function run()
    test_gap_column_generation_converges()
end

end # module ColumnGenerationE2eTests
