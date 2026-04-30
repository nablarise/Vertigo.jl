# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using JSON
using JuMP
using HiGHS
using MathOptInterface
using Vertigo

const MOI = MathOptInterface

struct GAPInstance
    name::String
    n_machines::Int
    n_tasks::Int
    cost::Matrix{Float64}
    weight::Matrix{Float64}
    capacity::Vector{Float64}
end

"""
    parse_gap_json(path::String) -> GAPInstance

Parse a GAPLIB JSON instance. Schema:
`{name, numcli, numserv, cost::Matrix, req::Matrix, cap::Vector}`
where `cost` and `req` are stored as `m × n` (machine × task).
"""
function parse_gap_json(path::String)::GAPInstance
    raw = JSON.parsefile(path)
    name = String(raw["name"])
    n = Int(raw["numcli"])
    m = Int(raw["numserv"])

    cost = Matrix{Float64}(undef, m, n)
    weight = Matrix{Float64}(undef, m, n)
    for k in 1:m, t in 1:n
        cost[k, t] = Float64(raw["cost"][k][t])
        weight[k, t] = Float64(raw["req"][k][t])
    end

    capacity = Vector{Float64}(undef, m)
    for k in 1:m
        capacity[k] = Float64(raw["cap"][k])
    end

    return GAPInstance(name, m, n, cost, weight, capacity)
end

"""
    build_gap_context(inst::GAPInstance; smoothing_alpha=0.0) -> ColGenLoggerWorkspace

Build a Dantzig-Wolfe column-generation workspace for `inst`. One subproblem
per machine (knapsack), one assignment constraint per task in the master.
"""
function build_gap_context(inst::GAPInstance; smoothing_alpha::Float64=0.0)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)
    @constraint(master_jump, assignment[t in T], 0 == 1)
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)
    @objective(master_jump, Min, 0)

    master_model = backend(master_jump)

    sp_models = Dict{PricingSubproblemId,Any}()
    sp_var_indices = Dict{PricingSubproblemId,Vector{MOI.VariableIndex}}()

    for k in K
        sp_jump = Model(HiGHS.Optimizer)
        set_silent(sp_jump)
        @variable(sp_jump, z[t in T], Bin)
        @constraint(
            sp_jump,
            sum(inst.weight[k, t] * z[t] for t in T) <= inst.capacity[k]
        )
        @objective(sp_jump, Min, sum(inst.cost[k, t] * z[t] for t in T))

        sp_models[PricingSubproblemId(k)] = backend(sp_jump)
        sp_var_indices[PricingSubproblemId(k)] = [index(z[t]) for t in T]
    end

    builder = DWReformulationBuilder{Tuple{Int,Int}}(minimize=true)

    for k in K
        add_subproblem!(builder, PricingSubproblemId(k), 0.0, 0.0, 1.0)
    end

    for k in K, t in T
        sp_var = sp_var_indices[PricingSubproblemId(k)][t]
        add_sp_variable!(builder, PricingSubproblemId(k), sp_var, inst.cost[k, t])
        cstr_idx = index(assignment[t])
        add_coupling_coefficient!(
            builder, PricingSubproblemId(k), sp_var, cstr_idx, 1.0
        )
        add_mapping!(builder, (k, t), PricingSubproblemId(k), sp_var)
    end

    for t in T
        add_coupling_constraint!(builder, index(assignment[t]), 1.0)
    end

    decomp = build(builder)

    pool = ColumnPool()
    conv_ub_map = Dict{PricingSubproblemId,TaggedCI}(
        PricingSubproblemId(k) => TaggedCI(index(conv_ub[k])) for k in K
    )
    conv_lb_map = Dict{PricingSubproblemId,TaggedCI}(
        PricingSubproblemId(k) => TaggedCI(index(conv_lb[k])) for k in K
    )

    set_models!(decomp, master_model, sp_models, conv_ub_map, conv_lb_map)

    config = ColGenConfig(
        smoothing_alpha=smoothing_alpha, silent=true
    )
    inner_ws = ColGenWorkspace(
        decomp, pool,
        Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        config
    )
    return inner_ws
end
