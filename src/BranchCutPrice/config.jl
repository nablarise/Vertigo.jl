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
- `cg_config::ColGenConfig`: Column generation configuration applied to the
  internal `ColGenWorkspace` built from the decomposition (default:
  `ColGenConfig()`).
- `time_limit::Float64`: Wall-clock limit in seconds of the whole search
  (default: `Inf`). It is checked between nodes and, through the column
  generation workspace, between column generation iterations; the
  restricted master heuristic gets at most the time left.
- `node_limit::Int`: Maximum nodes to explore (default: 10000).
- `rmp_heuristic_time_limit::Float64`: Time limit in seconds for restricted master
  IP heuristic at each node (default: 60.0).
- `rmp_use_heuristic::Bool`: Run the restricted master IP heuristic at each
  node to find feasible solutions (default: true).
- `node_select_strategy::TreeSearch.AbstractSearchStrategy`: Node selection strategy (default: `DepthFirstStrategy()`).
- `branching_strategy`: Branching strategy
  (default: `MostFractionalBranching()`).
- `cut_config::CutSeparationConfig`: Cut separator and the rounds of the
  cut-and-column-generation loop (default: `CutSeparationConfig()`, no
  cut separation).
- `tolerance::Float64`: Numerical tolerance (default: 1e-6).
- `log_level::Int`: BCP tree / branching logger verbosity (0 = off,
  1 = table, 2 = BaPCod-style verbose). Default: 0.
- `dot_file::Union{Nothing,String}`: Path for Graphviz `.dot` tree output
  (default: `nothing` — no dot file written).
"""
struct BranchCutPriceConfig
    cg_config::ColGen.ColGenConfig

    time_limit::Float64
    node_limit::Int

    rmp_heuristic_time_limit::Float64
    rmp_use_heuristic::Bool

    node_select_strategy::TreeSearch.AbstractSearchStrategy
    branching_strategy::AbstractBranchingStrategy
    cut_config::CutSeparationConfig

    tolerance::Float64

    log_level::Int
    dot_file::Union{Nothing,String}

    function BranchCutPriceConfig(;
        cg_config::ColGen.ColGenConfig=ColGen.ColGenConfig(),
        time_limit::Float64=Inf,
        node_limit::Int=10_000,
        rmp_heuristic_time_limit::Float64=60.0,
        rmp_use_heuristic::Bool=true,
        node_select_strategy::TreeSearch.AbstractSearchStrategy=TreeSearch.DepthFirstStrategy(),
        branching_strategy::AbstractBranchingStrategy=MostFractionalBranching(),
        cut_config::CutSeparationConfig=CutSeparationConfig(),
        tolerance::Float64=1e-6,
        log_level::Int=0,
        dot_file::Union{Nothing,String}=nothing,
    )
        return new(
            cg_config,
            time_limit,
            node_limit,
            rmp_heuristic_time_limit,
            rmp_use_heuristic,
            node_select_strategy,
            branching_strategy,
            cut_config,
            tolerance,
            log_level,
            dot_file,
        )
    end
end
