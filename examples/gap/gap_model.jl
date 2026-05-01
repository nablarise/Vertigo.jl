# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using JSON
using JuMP
using HiGHS
using Vertigo

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
    build_gap_model(inst::GAPInstance) -> DWReformulation

Build a flat JuMP model for `inst` and apply the Dantzig-Wolfe
decomposition: `x` and the per-machine `knapsack` constraint go to
subproblem `k`; the per-task `assign` constraint stays in the master.
"""
function build_gap_model(inst::GAPInstance)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[k in K, t in T], Bin)
    @constraint(model, assign[t in T], sum(x[k, t] for k in K) == 1)
    @constraint(
        model, knapsack[k in K],
        sum(inst.weight[k, t] * x[k, t] for t in T) <= inst.capacity[k]
    )
    @objective(
        model, Min, sum(inst.cost[k, t] * x[k, t] for k in K, t in T)
    )

    decomp, _ = @dantzig_wolfe model begin
        x[k, _]     => subproblem(k)
        assign[_]   => master()
        knapsack[k] => subproblem(k)
    end
    return decomp
end

"""
    build_col_gen_config(; smoothing_alpha=0.0) -> ColGenConfig

Construct the column-generation algorithm parameters used by the
benchmark.
"""
function build_col_gen_config(; smoothing_alpha::Float64=0.0)
    return ColGenConfig(
        smoothing_alpha=smoothing_alpha, silent=true
    )
end
