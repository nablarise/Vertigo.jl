# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    BreadthFirstNodeSelecStrategy <: AbstractNodeSelecStrategy

Breadth-first search. Uses a deque (FIFO) as the open-node container.
Nodes are explored level by level.
"""
struct BreadthFirstNodeSelecStrategy <: AbstractNodeSelecStrategy end

function init_container(::BreadthFirstNodeSelecStrategy, root::SearchNode)
    queue = Deque{typeof(root)}()
    push!(queue, root)
    return queue
end

select_node!(::BreadthFirstNodeSelecStrategy, queue::Deque) = popfirst!(queue)

function insert_node!(::BreadthFirstNodeSelecStrategy, queue::Deque, node::SearchNode)
    push!(queue, node)
    return nothing
end

is_empty_container(queue::Deque) = isempty(queue)
