# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module VertigoE2eTests

using Test
using Random
using JuMP
using HiGHS
using MathOptInterface
using ZipFile
const MOI = MathOptInterface

using Vertigo

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
    # Maximization GAP with all-negative costs, 3 machines × 30 jobs.
    # Costs vary by machine and task to create a non-trivial LP relaxation.
    n_machines = 3
    n_tasks = 30
    cost = [-(k + t) * 1.0 for k in 1:n_machines, t in 1:n_tasks]
    weight = fill(3.0, n_machines, n_tasks)
    capacity = fill(50.0, n_machines)  # 50/3 ≈ 16 tasks/machine, 48 > 30
    return GAPInstance(n_machines, n_tasks, cost, weight, capacity)
end

function gap_two_identical_machines()
    # Machines 1 and 2 are identical → symmetric degeneracy in LP relaxation.
    n_machines = 3
    n_tasks = 30
    cost = Matrix{Float64}(undef, n_machines, n_tasks)
    weight = Matrix{Float64}(undef, n_machines, n_tasks)
    for t in 1:n_tasks
        cost[1, t] = 2.0; cost[2, t] = 2.0; cost[3, t] = 3.0
        weight[1, t] = 3.0; weight[2, t] = 3.0; weight[3, t] = 4.0
    end
    capacity = fill(50.0, n_machines)
    return GAPInstance(n_machines, n_tasks, cost, weight, capacity)
end

function gap_three_identical_machines()
    # All three machines are identical → maximum symmetric degeneracy.
    n_machines = 3
    n_tasks = 30
    cost = fill(2.0, n_machines, n_tasks)
    weight = fill(3.0, n_machines, n_tasks)
    capacity = fill(50.0, n_machines)
    return GAPInstance(n_machines, n_tasks, cost, weight, capacity)
end

"""
    parse_gap_file(filepath::String) -> GAPInstance

Parse a GAP benchmark instance file into a `GAPInstance`.

Numbers are read as a flat whitespace-separated token stream regardless of
how they are laid out across lines (classes C/D use ~12 per line; class E
packs all jobs for one machine on a single line).
"""
function parse_gap_file(filepath::String)::GAPInstance
    tokens = split(read(filepath, String))
    idx = Ref(1)

    next_int() = (v = parse(Int, tokens[idx[]]); idx[] += 1; v)

    m = next_int()
    n = next_int()

    cost = Matrix{Float64}(undef, m, n)
    for i in 1:m, j in 1:n
        cost[i, j] = next_int()
    end

    weight = Matrix{Float64}(undef, m, n)
    for i in 1:m, j in 1:n
        weight[i, j] = next_int()
    end

    capacity = Vector{Float64}(undef, m)
    for i in 1:m
        capacity[i] = next_int()
    end

    return GAPInstance(m, n, cost, weight, capacity)
end

"""
    get_gap_instance_path(class::Char, agents::Int, jobs::Int) -> String

Extract a named GAP instance from its zip archive and return the path to the
extracted file in a temporary directory.

The zip archives are located at `test/data/gap/gap_{c,d,e}.zip` relative to
this file's directory. The entry name follows the convention `{class}{agents:02d}{jobs}`,
e.g. `('C', 5, 100)` → `"c05100"`.
"""
function get_gap_instance_path(class::Char, agents::Int, jobs::Int)::String
    lc = lowercase(class)
    zip_path = joinpath(@__DIR__, "..", "data", "gap", "gap_$(lc).zip")
    entry_name = "$(lc)$(lpad(agents, 2, '0'))$(jobs)"

    out_path = joinpath(mktempdir(), entry_name)

    zf = ZipFile.Reader(zip_path)
    found = false
    for f in zf.files
        if f.name == entry_name
            write(out_path, read(f))
            found = true
            break
        end
    end
    close(zf)

    found || error("Entry '$(entry_name)' not found in '$(zip_path)'")
    return out_path
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Build ColGenContext for a GAP instance
# ────────────────────────────────────────────────────────────────────────────────────────

function build_gap_context(inst::GAPInstance; smoothing_alpha::Float64=0.0)
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
    inner_ctx = ColGenContext(
        decomp,
        master_model,
        conv_ub_map,
        conv_lb_map,
        sp_models,
        pool,
        NonRobustCutManager{CstrId}(),
        Dict{Any,Any}(),
        Dict{Any,Any}(),
        Dict{Any,Any}();
        smoothing_alpha=smoothing_alpha
    )
    ctx = ColGenLoggerContext(inner_ctx)

    return ctx
end

function build_gap_context_max(inst::GAPInstance)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    # ── Master model ──────────────────────────────────────────────────────────
    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)

    @constraint(master_jump, assignment[t in T], 0 == 1)   # coupling: = 1
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)      # Σλ ≥ 0
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)      # Σλ ≤ 1
    @objective(master_jump, Max, 0)                          # maximization sense

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

    builder = DecompositionBuilder{Int,SpVar,Tuple{Int,Int},CstrId,Nothing}(minimize=false)

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
    inner_ctx = ColGenContext(
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
    ctx = ColGenLoggerContext(inner_ctx)

    return ctx
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
# Tests
# ────────────────────────────────────────────────────────────────────────────────────────

function test_gap_column_generation_converges()
    @testset "[gap] column generation converges (2 machines, 7 tasks)" begin
        inst = random_gap_instance(2, 7)
        ctx = build_gap_context(inst)

        output = run_column_generation(ctx)

        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
    end
end

function test_gap_benchmark_instances()
    for (class, agents, jobs, expected_bound) in BENCHMARK_INSTANCES
        label = "$(uppercase(class)) $(agents)×$(jobs)"
        @testset "[gap][$(label)] dual bound ≈ $(expected_bound)" begin
            filepath = get_gap_instance_path(class, agents, jobs)
            inst = parse_gap_file(filepath)
            ctx = build_gap_context(inst)

            output = run_column_generation(ctx)

            @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
            @test abs(output.incumbent_dual_bound - expected_bound) <= 1.0
        end
    end
end

function test_gap_infeasible_master()
    @testset "[gap] infeasible master (3 machines, 30 jobs, max 27 coverable)" begin
        inst = gap_infeasible_master()
        ctx  = build_gap_context(inst)
        output = run_column_generation(ctx)
        # Phase0 converges with art vars → Phase1 confirms infeasibility
        @test output.status == master_infeasible
    end
end

function test_gap_infeasible_subproblem()
    @testset "[gap] infeasible subproblem (machine 1 capacity = -1)" begin
        inst = gap_infeasible_subproblem()
        ctx = build_gap_context(inst)
        output = run_column_generation(ctx)
        @test output.status == subproblem_infeasible
    end
end

function test_gap_maximization()
    @testset "[gap] maximization converges (3 machines, 30 jobs, negative costs)" begin
        inst = gap_maximization()
        ctx = build_gap_context_max(inst)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
    end
end

function test_gap_two_identical_machines()
    @testset "[gap] two identical machines (symmetric degeneracy)" begin
        inst = gap_two_identical_machines()
        ctx = build_gap_context(inst)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
    end
end

function test_gap_three_identical_machines()
    @testset "[gap] three identical machines (maximum symmetry)" begin
        inst = gap_three_identical_machines()
        ctx = build_gap_context(inst)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
    end
end

function test_gap_wentges_smoothing()
    @testset "[gap] Wentges smoothing converges (2 machines, 7 tasks, α=0.5)" begin
        inst = random_gap_instance(2, 7)
        ctx = build_gap_context(inst; smoothing_alpha=0.5)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
    end
end

function test_gap_wentges_smoothing_larger()
    @testset "[gap] Wentges smoothing converges (3 machines, 30 tasks, α=0.5)" begin
        inst = gap_two_identical_machines()
        ctx = build_gap_context(inst; smoothing_alpha=0.5)
        output = run_column_generation(ctx)
        @test output.status == optimal
        @test abs(output.master_lp_obj - output.incumbent_dual_bound) <= 1e-4
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Branch-and-price e2e tests
# ────────────────────────────────────────────────────────────────────────────────────────

include("test_branch_and_price_e2e.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function run()
    test_gap_column_generation_converges()
    #test_gap_benchmark_instances()
    test_gap_infeasible_master()
    test_gap_infeasible_subproblem()
    test_gap_maximization()
    test_gap_two_identical_machines()
    test_gap_three_identical_machines()
    test_gap_wentges_smoothing()
    test_gap_wentges_smoothing_larger()

    for i in 1:10
        println("---------")
    end
    test_bp_gap_a_instances()
end

end # module VertigoE2eTests
