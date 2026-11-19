# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    AbstractBestFirstStrategy <: AbstractSearchStrategy

Abstract type for best-first search strategies.
Subtypes must implement `get_priority(strategy, node) -> Float64`.
Lower priority values are explored first in a minimization, higher ones in a
maximization.
"""
abstract type AbstractBestFirstStrategy <: AbstractSearchStrategy end

"""
    get_priority(strategy::AbstractBestFirstStrategy, node::SearchNode) -> Float64

Return the priority value for `node`, in the units of the objective: lower
values are explored first in a minimization, higher ones in a maximization.
"""
function get_priority end

"""
    DualBoundBestFirstStrategy <: AbstractBestFirstStrategy

Best-first search using the dual bound as priority.
The node with the most promising dual bound is explored first: the smallest in
a minimization, the largest in a maximization.
"""
struct DualBoundBestFirstStrategy <: AbstractBestFirstStrategy end

get_priority(::DualBoundBestFirstStrategy, node::SearchNode) = node.dual_bound

_priority_order(minimize) = minimize ? Base.Order.Forward : Base.Order.Reverse

# Container: PriorityQueue, a min-heap in a minimization, a max-heap otherwise.
function init_container(strategy::AbstractBestFirstStrategy, root::SearchNode, minimize)
    pq = PriorityQueue{typeof(root), Float64}(_priority_order(minimize))
    enqueue!(pq, root, get_priority(strategy, root))
    return pq
end

function select_node!(::AbstractBestFirstStrategy, pq::PriorityQueue)
    return dequeue!(pq)
end

function insert_node!(strategy::AbstractBestFirstStrategy, pq::PriorityQueue, node::SearchNode)
    enqueue!(pq, node, get_priority(strategy, node))
    return
end

is_empty_container(pq::PriorityQueue) = isempty(pq)
