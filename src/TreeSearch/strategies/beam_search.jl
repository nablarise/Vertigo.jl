# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    BeamSearchStrategy{S<:AbstractBestFirstStrategy} <: AbstractSearchStrategy

Beam search: explores the tree level by level, keeping only the `max_width`
best nodes at each depth level. Requires an inner best-first strategy to
determine node priorities.

Unlike the previous implementation, this does NOT wrap nodes. The beam width
control is handled entirely through the container management.
"""
struct BeamSearchStrategy{S<:AbstractBestFirstStrategy} <: AbstractSearchStrategy
    inner_strategy::S
    max_width::Int
end

"""
    BeamSearchContainer{N}

Two-queue container for beam search: one for the current depth level and one
for the next depth level. Nodes are transferred between queues when the
current level is exhausted or the beam width is reached.
"""
mutable struct BeamSearchContainer{N,O<:Base.Order.Ordering}
    current_queue::PriorityQueue{N, Float64, O}
    next_queue::PriorityQueue{N, Float64, O}
    nodes_at_current_depth::Int
    max_width::Int
end

function init_container(strategy::BeamSearchStrategy, root::SearchNode, minimize)
    N = typeof(root)
    order = _priority_order(minimize)
    current = PriorityQueue{N, Float64}(order)
    next = PriorityQueue{N, Float64}(order)
    enqueue!(current, root, get_priority(strategy.inner_strategy, root))
    return BeamSearchContainer(current, next, 0, strategy.max_width)
end

function select_node!(strategy::BeamSearchStrategy, container::BeamSearchContainer)
    # If current queue is exhausted or beam width reached, swap queues
    if isempty(container.current_queue) || container.nodes_at_current_depth >= container.max_width
        dropped = container.current_queue
        container.current_queue = container.next_queue
        container.next_queue = empty!(dropped)
        container.nodes_at_current_depth = 0
    end

    container.nodes_at_current_depth += 1
    return dequeue!(container.current_queue)
end

function insert_node!(strategy::BeamSearchStrategy, container::BeamSearchContainer, node::SearchNode)
    enqueue!(container.next_queue, node, get_priority(strategy.inner_strategy, node))
    return
end

function is_empty_container(container::BeamSearchContainer)
    # The current level is exhausted if it's empty or the beam width has been reached.
    # If so, only the next level matters.
    exhausted_current = isempty(container.current_queue) ||
                        container.nodes_at_current_depth >= container.max_width
    return exhausted_current && isempty(container.next_queue)
end
