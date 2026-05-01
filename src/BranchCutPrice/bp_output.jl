# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    BPOutput

Result of a branch-and-price run.

# Fields
- `status::Symbol`: `:optimal`, `:infeasible`, or `:node_limit`.
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
