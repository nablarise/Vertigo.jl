# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    DepthFirstNodeSelecStrategy <: AbstractNodeSelecStrategy

Depth-first search. Uses a stack (LIFO) as the open-node container.
Children are pushed in reverse order so that the first child is explored first.
"""
struct DepthFirstNodeSelecStrategy <: AbstractNodeSelecStrategy end

function init_container(::DepthFirstNodeSelecStrategy, root::SearchNode)
    stack = Stack{typeof(root)}()
    push!(stack, root)
    return stack
end

select_node!(::DepthFirstNodeSelecStrategy, stack::Stack) = pop!(stack)

function insert_node!(::DepthFirstNodeSelecStrategy, stack::Stack, node::SearchNode)
    push!(stack, node)
    return nothing
end

is_empty_container(stack::Stack) = isempty(stack)

"""
    insert_children!(::DepthFirstNodeSelecStrategy, stack, children)

Override for DFS: insert children in reverse order so the first child
in the iterable is popped (explored) first.
"""
function insert_children!(::DepthFirstNodeSelecStrategy, stack::Stack, children)
    collected = collect(children)
    for i in length(collected):-1:1
        push!(stack, collected[i])
    end
    return nothing
end
