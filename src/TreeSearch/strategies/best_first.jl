# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    AbstractBestFirstStrategy <: AbstractSearchStrategy

Abstract type for best-first search strategies. The open nodes are kept in a
min-heap keyed by a priority, and the node with the lowest priority is selected
first.

Subtypes must implement [`get_priority`](@ref). A strategy is an ordinary
struct, so it may hold parameters and state (a random number generator, say);
`get_priority` is called once per inserted node, and a strategy that mutates
its state is shared by every search that uses it.

```julia
struct DeepestFirst <: AbstractBestFirstStrategy end
get_priority(::DeepestFirst, node::NodeView) = -node.depth
```
"""
abstract type AbstractBestFirstStrategy <: AbstractSearchStrategy end

"""
    get_priority(strategy::AbstractBestFirstStrategy, node::NodeView) -> Float64

Return the priority of `node`, a read-only [`NodeView`](@ref).

- **Lower is first.** The node with the lowest priority is selected first; use the
  opposite of a quantity to maximise it.
- **Ties** are broken in an unspecified order; do not rely on insertion order.
- **Computed once,** when the node is inserted in the open nodes, and never
  recomputed: the open nodes are not re-ranked when the incumbent or the dual
  bound change.
- **Node only.** The function sees the node, not the search space.

TODO: give the priority function read access to the search space (incumbent,
global dual bound, number of nodes) once the use cases are known; the open
nodes would still not be re-ranked.
"""
function get_priority end

"""
    DualBoundBestFirstStrategy <: AbstractBestFirstStrategy

Best-first search using the dual bound as priority.
Nodes with smaller dual bounds are explored first (standard best-first).
"""
struct DualBoundBestFirstStrategy <: AbstractBestFirstStrategy end

get_priority(::DualBoundBestFirstStrategy, node::NodeView) = node.dual_bound

# Container: PriorityQueue (min-heap by default)
function init_container(strategy::AbstractBestFirstStrategy, root::SearchNode)
    pq = PriorityQueue{typeof(root),Float64}()
    enqueue!(pq, root, get_priority(strategy, NodeView(root)))
    return pq
end

function select_node!(::AbstractBestFirstStrategy, pq::PriorityQueue)
    return dequeue!(pq)
end

function insert_node!(
    strategy::AbstractBestFirstStrategy, pq::PriorityQueue, node::SearchNode
)
    enqueue!(pq, node, get_priority(strategy, NodeView(node)))
    return nothing
end

is_empty_container(pq::PriorityQueue) = isempty(pq)
