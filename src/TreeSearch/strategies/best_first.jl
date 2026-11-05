# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    AbstractBestFirstStrategy <: AbstractSearchStrategy

Abstract type for best-first search strategies.
Subtypes must implement `get_priority(strategy, node::NodeView) -> Float64`.
Lower priority values are explored first (min-heap).
"""
abstract type AbstractBestFirstStrategy <: AbstractSearchStrategy end

"""
    get_priority(strategy::AbstractBestFirstStrategy, node::NodeView) -> Float64

Return the priority value for `node`, a read-only [`NodeView`](@ref). Lower values
are explored first.

The priority is computed once, when the node is inserted in the open nodes, and is
not recomputed afterwards.

TODO: the priority function sees the node only. Give it read access to the search
space (incumbent, global dual bound, number of nodes) once the use cases are known;
note that the open nodes are not re-ranked when the space changes.
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
