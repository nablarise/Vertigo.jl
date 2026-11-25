# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    BPOutput

Result of a branch-and-price run.

# Fields
- `status::Symbol`: `:optimal`, `:infeasible`, `:node_limit`,
  `:time_limit` or `:branching_incomplete`. The last one means the search
  ended with nodes it could not branch: their projection onto the original
  variables is integral but their columns are not, and their bound is
  neither closed nor pruned by the incumbent. `best_dual_bound` covers
  them. A node or time limit takes precedence over it.
- `incumbent`: Best integer-feasible solution found, or `nothing`.
- `best_dual_bound::Float64`: Best dual bound across all nodes.
- `nodes_explored::Int`: Total number of nodes evaluated.
- `root_lp_value::Union{Float64,Nothing}`: Master LP objective at the
  root node, or `nothing` if the root LP was infeasible or never solved.
"""
struct BPOutput
    status::Symbol
    incumbent::Union{Nothing,ColGen.MasterIpPrimalSol}
    best_dual_bound::Float64
    nodes_explored::Int
    root_lp_value::Union{Float64,Nothing}
end
