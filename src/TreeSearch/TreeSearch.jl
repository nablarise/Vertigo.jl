# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module TreeSearch

using DataStructures
using Printf

# Node status enum
include("node_status.jl")

# Search node and tree navigation
include("node.jl")

# Abstract types and interface declarations
include("interface.jl")

# Generic search loop
include("search_loop.jl")

# Logger context
include("logger.jl")

# Strategies
include("strategies/depth_first.jl")
include("strategies/best_first.jl")
include("strategies/breadth_first.jl")
include("strategies/beam_search.jl")
include("strategies/limited_discrepancy.jl")

# Public API
export AbstractSearchStrategy, AbstractSearchSpace, AbstractNodeEvaluator
export NodeStatus, CUTOFF, FEASIBLE, BRANCH
export SearchNode, NodeIdCounter, root_node, child_node, is_root
export find_common_ancestor, transition_to!, collect_path_from_ancestor
export search, new_root, stop, output, transition!, branch!, evaluate!
export on_feasible_solution!, prune!
export init_container, select_node!, insert_node!, insert_children!, is_empty_container
export DepthFirstStrategy
export BreadthFirstStrategy
export AbstractBestFirstStrategy, DualBoundBestFirstStrategy, get_priority
export BeamSearchStrategy
export LimitedDiscrepancyStrategy, DiscrepancyData, filter_by_discrepancy
export TreeSearchLoggerContext
export ts_incumbent_value, ts_best_dual_bound, ts_is_minimization
export ts_nodes_explored, ts_search_status_message
export ts_open_node_count, ts_total_columns, ts_active_columns
export ts_total_cuts, ts_branching_description

end # module TreeSearch
