# Copyright (c) 2026 Nablarise. All rights reserved.
# Originally created by: Guillaume Marques <guillaume@nablarise.com>
# Project: Vertigo.jl
# Date: 2026-04
# Generated with: Claude Opus 4.6
# SPDX-License-Identifier: MIT

"""
    BranchCutPriceConfig

Plain data holder for user-defined branch-cut-and-price parameters.
Cheap to construct, copy, and compare.

# Fields
- `colgen::ColGenConfig`: Column generation configuration applied to the
  internal `ColGenWorkspace` built from the decomposition (default:
  `ColGenConfig()`).
- `cg_log_level::Int`: CG-iteration logger verbosity. `0` uses a bare
  `ColGenWorkspace`; values `> 0` wrap it in a `ColGenLoggerWorkspace`
  with that level (default: `0`).
- `strategy`: Tree search strategy (default: `DepthFirstStrategy()`).
- `node_limit::Int`: Maximum nodes to explore (default: 10000).
- `tol::Float64`: Numerical tolerance (default: 1e-6).
- `rmp_time_limit::Float64`: Time limit in seconds for restricted master
  IP heuristic at each node (default: 60.0).
- `rmp_heuristic::Bool`: Run the restricted master IP heuristic at each
  node to find feasible solutions (default: true).
- `separator`: Robust cut separator (default: `nothing`).
- `max_cut_rounds::Int`: Maximum cut separation rounds per node
  (default: 0).
- `min_gap_improvement::Float64`: Minimum relative gap improvement to
  continue cut rounds (default: 0.01).
- `branching_strategy`: Branching strategy
  (default: `MostFractionalBranching()`).
- `log_level::Int`: BCP tree / branching logger verbosity (0 = off,
  1 = table, 2 = BaPCod-style verbose). Default: 0.
- `dot_file::Union{Nothing,String}`: Path for Graphviz `.dot` tree output
  (default: `nothing` — no dot file written).
"""
struct BranchCutPriceConfig
    colgen::ColGen.ColGenConfig
    cg_log_level::Int
    strategy::Any
    node_limit::Int
    tol::Float64
    rmp_time_limit::Float64
    rmp_heuristic::Bool
    separator::Union{Nothing,AbstractCutSeparator}
    max_cut_rounds::Int
    min_gap_improvement::Float64
    branching_strategy::AbstractBranchingStrategy
    log_level::Int
    dot_file::Union{Nothing,String}
    function BranchCutPriceConfig(;
        colgen::ColGen.ColGenConfig = ColGen.ColGenConfig(),
        cg_log_level::Int = 0,
        strategy = TreeSearch.DepthFirstStrategy(),
        node_limit::Int = 10_000,
        tol::Float64 = 1e-6,
        rmp_time_limit::Float64 = 60.0,
        rmp_heuristic::Bool = true,
        separator::Union{Nothing,AbstractCutSeparator} = nothing,
        max_cut_rounds::Int = 0,
        min_gap_improvement::Float64 = 0.01,
        branching_strategy::AbstractBranchingStrategy =
            MostFractionalBranching(),
        log_level::Int = 0,
        dot_file::Union{Nothing,String} = nothing
    )
        new(colgen, cg_log_level, strategy, node_limit, tol,
            rmp_time_limit, rmp_heuristic, separator, max_cut_rounds,
            min_gap_improvement, branching_strategy,
            log_level, dot_file)
    end
end
