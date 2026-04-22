# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    DepthFirstStrategy <: AbstractSearchStrategy

Depth-first search. Uses a stack (LIFO) as the open-node container.
Children are pushed in reverse order so that the first child is explored first.
"""
struct DepthFirstStrategy <: AbstractSearchStrategy end

function init_container(::DepthFirstStrategy, root::SearchNode)
    stack = Stack{typeof(root)}()
    push!(stack, root)
    return stack
end

select_node!(::DepthFirstStrategy, stack::Stack) = pop!(stack)

function insert_node!(::DepthFirstStrategy, stack::Stack, node::SearchNode)
    push!(stack, node)
    return
end

is_empty_container(stack::Stack) = isempty(stack)

"""
    insert_children!(::DepthFirstStrategy, stack, children)

Override for DFS: insert children in reverse order so the first child
in the iterable is popped (explored) first.
"""
function insert_children!(::DepthFirstStrategy, stack::Stack, children)
    collected = collect(children)
    for i in length(collected):-1:1
        push!(stack, collected[i])
    end
    return
end
