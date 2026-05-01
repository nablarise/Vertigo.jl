# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Dates
using JSON
using Printf
using Vertigo

include("gap_model.jl")

struct BenchConfig
    node_limit::Int
    smoothing_alpha::Float64
    rmp_time_limit::Float64   # per-node IP heuristic time cap (seconds)
end

mutable struct BenchResult
    instance::String
    n::Int
    m::Int
    status::Symbol
    wall_time_s::Float64
    primal_bound::Union{Float64,Nothing}
    dual_bound::Union{Float64,Nothing}
    gap_pct::Union{Float64,Nothing}
    opt_known::Union{Float64,Nothing}
    nb_nodes::Int
    nb_columns::Union{Int,Nothing}        # blocked on issue #60
    nb_rmp_iters::Union{Int,Nothing}      # blocked on issue #61
    root_lp_value::Union{Float64,Nothing} # blocked on issue #62
end

const METADATA_FILE = joinpath(@__DIR__, "instances_metadata.json")

function load_metadata()
    return isfile(METADATA_FILE) ? JSON.parsefile(METADATA_FILE) : Dict{String,Any}()
end

function lookup_opt(meta, name)
    haskey(meta, name) || return nothing
    v = get(meta[name], "opt", nothing)
    return v === nothing ? nothing : Float64(v)
end

function compute_gap(primal, dual)
    (primal === nothing || dual === nothing) && return nothing
    (isfinite(primal) && isfinite(dual)) || return nothing
    abs(primal) < 1e-9 && return nothing
    return 100.0 * abs(primal - dual) / abs(primal)
end

# Coerce non-finite floats to nothing for JSON serialization.
finite_or_nothing(x) = (x isa Real && isfinite(x)) ? x : nothing
finite_or_nothing(::Nothing) = nothing

"""
    run_one(json_path, cfg, meta) -> BenchResult

Solve a single GAP instance via branch-cut-price and return its metrics.
"""
function run_one(json_path::String, cfg::BenchConfig, meta)::BenchResult
    inst = parse_gap_json(json_path)
    opt = lookup_opt(meta, inst.name)

    local output
    elapsed = @elapsed begin
        try
            decomp = build_gap_model(inst)
            colgen_config = ColGenConfig(
                smoothing_alpha=cfg.smoothing_alpha, silent=true
            )
            ws = ColGenWorkspace(decomp, colgen_config)
            bcp_ctx = BranchCutPriceContext(
                ws;
                node_limit = cfg.node_limit,
                rmp_time_limit = cfg.rmp_time_limit,
                log_level = 0
            )
            output = run_branch_and_price(bcp_ctx)
        catch err
            @warn "instance $(inst.name) failed" exception=(err, catch_backtrace())
            output = nothing
        end
    end

    if output === nothing
        return BenchResult(
            inst.name, inst.n_tasks, inst.n_machines,
            :error, elapsed,
            nothing, nothing, nothing, opt,
            0, nothing, nothing, nothing
        )
    end

    primal = output.incumbent === nothing ? nothing : output.incumbent.obj_value
    dual = output.best_dual_bound
    gap = compute_gap(primal, dual)

    return BenchResult(
        inst.name, inst.n_tasks, inst.n_machines,
        output.status, elapsed,
        primal, dual, gap, opt,
        output.nodes_explored,
        nothing, nothing, nothing
    )
end

# ── Output writers ──────────────────────────────────────────────────────

const CSV_HEADER = [
    "instance", "n", "m", "status", "wall_time_s",
    "primal_bound", "dual_bound", "gap_pct", "opt_known",
    "nb_nodes", "nb_columns", "nb_rmp_iters", "root_lp_value"
]

cell(::Nothing) = ""
cell(x::Float64) = @sprintf("%.6f", x)
cell(x::Symbol) = String(x)
cell(x) = string(x)

function write_csv(results::Vector{BenchResult}, path::String)
    open(path, "w") do io
        println(io, join(CSV_HEADER, ","))
        for r in results
            row = [
                r.instance, r.n, r.m, r.status, r.wall_time_s,
                r.primal_bound, r.dual_bound, r.gap_pct, r.opt_known,
                r.nb_nodes, r.nb_columns, r.nb_rmp_iters, r.root_lp_value
            ]
            println(io, join(cell.(row), ","))
        end
    end
end

function result_to_dict(r::BenchResult)
    return Dict(
        "instance" => r.instance, "n" => r.n, "m" => r.m,
        "status" => String(r.status), "wall_time_s" => r.wall_time_s,
        "primal_bound" => finite_or_nothing(r.primal_bound),
        "dual_bound" => finite_or_nothing(r.dual_bound),
        "gap_pct" => r.gap_pct, "opt_known" => r.opt_known,
        "nb_nodes" => r.nb_nodes, "nb_columns" => r.nb_columns,
        "nb_rmp_iters" => r.nb_rmp_iters,
        "root_lp_value" => finite_or_nothing(r.root_lp_value)
    )
end

function write_json(results::Vector{BenchResult}, meta::Dict, path::String)
    payload = Dict(
        "metadata" => meta,
        "results" => [result_to_dict(r) for r in results]
    )
    open(path, "w") do io
        JSON.print(io, payload, 2)
    end
end

function print_table(results::Vector{BenchResult}, io::IO=stdout)
    @printf(io, "%-18s %4s %3s %-12s %10s %12s %12s %8s %8s\n",
        "instance", "n", "m", "status", "time(s)",
        "primal", "dual", "gap%", "nodes")
    println(io, "-"^100)
    for r in results
        primal = r.primal_bound === nothing ? "—" : @sprintf("%.2f", r.primal_bound)
        dual = r.dual_bound === nothing ? "—" : @sprintf("%.2f", r.dual_bound)
        gap = r.gap_pct === nothing ? "—" : @sprintf("%.2f", r.gap_pct)
        @printf(io, "%-18s %4d %3d %-12s %10.3f %12s %12s %8s %8d\n",
            r.instance, r.n, r.m, String(r.status), r.wall_time_s,
            primal, dual, gap, r.nb_nodes)
    end
end

# ── Main loop ───────────────────────────────────────────────────────────

function git_commit_short()
    try
        return strip(read(`git rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function run_bench(
    instance_paths::Vector{String}, cfg::BenchConfig, mode::String;
    results_dir::String = joinpath(@__DIR__, "results")
)
    mkpath(results_dir)
    meta_lookup = load_metadata()

    println("== Vertigo GAP benchmark ($(mode)) ==")
    println("instances: $(length(instance_paths))   node_limit: $(cfg.node_limit)   " *
            "smoothing: $(cfg.smoothing_alpha)")
    println()

    results = BenchResult[]
    for (i, path) in enumerate(instance_paths)
        @printf("[%d/%d] %-30s ... ", i, length(instance_paths), basename(path))
        flush(stdout)
        r = run_one(path, cfg, meta_lookup)
        push!(results, r)
        @printf("%s in %.2fs\n", String(r.status), r.wall_time_s)
    end

    println()
    print_table(results)

    ts = Dates.format(now(UTC), "yyyymmddTHHMMSSZ")
    csv_path = joinpath(results_dir, "$(ts)-$(mode).csv")
    json_path = joinpath(results_dir, "$(ts)-$(mode).json")

    run_meta = Dict(
        "timestamp" => ts,
        "mode" => mode,
        "git_commit" => git_commit_short(),
        "julia_version" => string(VERSION),
        "config" => Dict(
            "node_limit" => cfg.node_limit,
            "smoothing_alpha" => cfg.smoothing_alpha,
            "rmp_time_limit" => cfg.rmp_time_limit
        )
    )
    write_csv(results, csv_path)
    write_json(results, run_meta, json_path)

    println()
    println("results: $(csv_path)")
    println("         $(json_path)")
    return results
end
