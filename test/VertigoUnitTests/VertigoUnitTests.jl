# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module VertigoUnitTests

using Test
using Random
using JuMP
using HiGHS
using MathOptInterface
const MOI = MathOptInterface

using Vertigo
using Vertigo.ColGen: MasterPrimalSolution, PrimalMoiSolution,
    check_primal_ip_feasibility!, update_inc_primal_sol!, Phase0, Phase1, Phase2,
    get_column_sp_id, get_column_cost, get_column_solution
using Vertigo.TreeSearch
using Vertigo.MathOptState
using Vertigo.Reformulation: get_primal_solution
using Vertigo.Branching: find_fractional_variables,
    BranchingCandidate, MostFractionalRule, LeastFractionalRule,
    select_candidates, MostFractionalBranching, select_branching_variable,
    bp_master_model, most_fractional_original_variable

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
function build_gap_context(inst::GAPInstance; max_cg_iterations::Int=1000)
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
    sp_models = Dict{PricingSubproblemId,Any}()
    sp_var_indices = Dict{PricingSubproblemId,Vector{MOI.VariableIndex}}()

    for k in K
        sp_jump = Model(HiGHS.Optimizer)
        set_silent(sp_jump)

        @variable(sp_jump, z[t in T], Bin)
        @constraint(sp_jump, sum(inst.weight[k, t] * z[t] for t in T) <= inst.capacity[k])
        @objective(sp_jump, Min, sum(inst.cost[k, t] * z[t] for t in T))

        sp_models[PricingSubproblemId(k)] = backend(sp_jump)
        sp_var_indices[PricingSubproblemId(k)] = [index(z[t]) for t in T]
    end

    # ── Build Decomposition ───────────────────────────────────────────────────
    SpVar = MOI.VariableIndex
    CstrId = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}

    builder = DWReformulationBuilder{Tuple{Int,Int}}(minimize=true)

    for k in K
        add_subproblem!(builder, PricingSubproblemId(k), 0.0, 0.0, 1.0)
    end

    for k in K
        for t in T
            sp_var = sp_var_indices[PricingSubproblemId(k)][t]
            add_sp_variable!(builder, PricingSubproblemId(k), sp_var, inst.cost[k, t])
            cstr_idx = index(assignment[t])
            add_coupling_coefficient!(builder, PricingSubproblemId(k), sp_var, cstr_idx, 1.0)
            add_mapping!(builder, (k, t), PricingSubproblemId(k), sp_var)
        end
    end

    for t in T
        add_coupling_constraint!(builder, index(assignment[t]), 1.0)
    end

    decomp = build(builder)

    # ── Column pool ───────────────────────────────────────────────────────────
    pool = ColumnPool()

    # ── Convexity constraint indices ──────────────────────────────────────────
    conv_ub_map = Dict{PricingSubproblemId,TaggedCI}(PricingSubproblemId(k) => TaggedCI(index(conv_ub[k])) for k in K)
    conv_lb_map = Dict{PricingSubproblemId,TaggedCI}(PricingSubproblemId(k) => TaggedCI(index(conv_lb[k])) for k in K)

    set_models!(decomp, master_model, sp_models, conv_ub_map, conv_lb_map)

    # ── Build context ─────────────────────────────────────────────────────────
    ctx = ColGenContext(
        decomp,
        pool,
        Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        Dict{TaggedCI,MOI.VariableIndex}();
        max_cg_iterations=max_cg_iterations
    )

    return ctx
end

include("colgen/test_gap.jl")
include("colgen/ip_management_tests.jl")
include("colgen/test_branch_and_price.jl")
include("colgen/test_stabilization.jl")
include("colgen/test_setup_reformulation.jl")
include("colgen/test_insert_columns.jl")
include("colgen/test_build_cut_saf.jl")
include("colgen/test_column_pool.jl")
include("treesearch/test_node.jl")
include("treesearch/test_search_loop.jl")
include("treesearch/test_strategies.jl")
include("treesearch/test_tree_search_logger.jl")
include("test_local_cut_tracker.jl")
include("test_lp_basis_tracker.jl")
include("test_column_tracker.jl")
include("test_cut_pool_tracker.jl")
include("colgen/test_max_cg_iterations.jl")
include("colgen/test_branching_strategy.jl")
include("colgen/test_lp_probe.jl")
include("colgen/test_strong_branching.jl")
include("colgen/test_pseudocosts.jl")
include("colgen/test_kernel.jl")
include("dml/test_dml.jl")

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function run()
    test_gap_decomposition_builder()
    test_gap_column_pool_populated()
    test_gap_lp_dual_bound_matches_primal()
    test_ip_management()
    test_treesearch_node()
    test_treesearch_search_loop()
    test_treesearch_strategies()
    test_tree_search_logger()
    test_local_cut_tracker()
    test_lp_basis_tracker()
    test_column_tracker()
    test_cut_pool_tracker()
    test_max_cg_iterations()
    test_branching_strategy()
    test_lp_probe()
    test_strong_branching()
    test_pseudocosts()
    test_kernel()
    test_branch_and_price()
    test_stabilization()
    test_setup_reformulation()
    test_insert_columns()
    test_build_cut_saf()
    test_column_pool()
    test_dml()
end

end # module VertigoUnitTests
