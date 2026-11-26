# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    BPStatus

Why a branch-and-price run ended.

- `OPTIMAL`: the incumbent's value matches the best dual bound.
- `INFEASIBLE`: the tree was fully explored without an incumbent.
- `NODE_LIMIT`: the node limit was reached first.
- `TIME_LIMIT`: the time limit was reached first.
- `BRANCHING_INCOMPLETE`: the search ended with nodes it could not branch:
  their projection onto the original variables is integral but their
  columns are not, and their bound is neither closed nor pruned by the
  incumbent. `best_dual_bound` covers them. A node or time limit takes
  precedence over it.

The values are not exported from `Vertigo`, because JuMP exports
`OPTIMAL`, `INFEASIBLE`, `NODE_LIMIT` and `TIME_LIMIT`: write
`Vertigo.OPTIMAL`.
"""
@enum BPStatus begin
    OPTIMAL
    INFEASIBLE
    NODE_LIMIT
    TIME_LIMIT
    BRANCHING_INCOMPLETE
end

"""
    BPOutput

Result of a branch-and-price run.

# Fields
- `status::BPStatus`: Why the run ended; see [`BPStatus`](@ref).
- `incumbent`: Best integer-feasible solution found, or `nothing`.
- `best_dual_bound::Float64`: Best dual bound across all nodes.
- `nodes_explored::Int`: Total number of nodes evaluated.
- `root_lp_value::Union{Float64,Nothing}`: Master LP objective at the
  root node, or `nothing` if the root LP was infeasible or never solved.
"""
struct BPOutput
    status::BPStatus
    incumbent::Union{Nothing,ColGen.MasterIpPrimalSol}
    best_dual_bound::Float64
    nodes_explored::Int
    root_lp_value::Union{Float64,Nothing}
end
