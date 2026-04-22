# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    BreadthFirstStrategy <: AbstractSearchStrategy

Breadth-first search. Uses a deque (FIFO) as the open-node container.
Nodes are explored level by level.
"""
struct BreadthFirstStrategy <: AbstractSearchStrategy end

function init_container(::BreadthFirstStrategy, root::SearchNode)
    queue = Deque{typeof(root)}()
    push!(queue, root)
    return queue
end

select_node!(::BreadthFirstStrategy, queue::Deque) = popfirst!(queue)

function insert_node!(::BreadthFirstStrategy, queue::Deque, node::SearchNode)
    push!(queue, node)
    return
end

is_empty_container(queue::Deque) = isempty(queue)
