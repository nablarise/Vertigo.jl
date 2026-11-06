# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    PlungingNodeSelecStrategy{S<:AbstractBestFirstNodeSelecStrategy} <: AbstractNodeSelecStrategy

Plunging (diving) search: after a node is branched, the search goes on with its
first child, as depth first search does, until a node is closed or has no child.
It then backtracks to the open node with the lowest priority according to
`inner_strategy`, and plunges again from there.

The plunges find feasible solutions early, and the best-first backtracking keeps
the global dual bound moving, which is the usual choice in branch-and-price.

# Fields
- `inner_strategy`: the best-first strategy that ranks the open nodes on
  backtracking (for example `DualBoundBestFirstNodeSelecStrategy()`).

# Convention
The first child returned by `branch!` is the one the search plunges into, as for
[`LimitedDiscrepancyNodeSelecStrategy`](@ref); the other children wait among the open
nodes.
"""
struct PlungingNodeSelecStrategy{S<:AbstractBestFirstNodeSelecStrategy} <:
       AbstractNodeSelecStrategy
    inner_strategy::S
end

function PlungingNodeSelecStrategy()
    return PlungingNodeSelecStrategy(DualBoundBestFirstNodeSelecStrategy())
end

"""
    PlungingContainer{N}

Open nodes of a plunging search: the `dive` node the search goes on with (if
any), and the other open nodes in a priority queue.
"""
mutable struct PlungingContainer{N}
    dive::Union{Nothing,N}
    open::PriorityQueue{N,Float64}
end

function init_container(strategy::PlungingNodeSelecStrategy, root::SearchNode)
    open = PriorityQueue{typeof(root),Float64}()
    enqueue!(open, root, get_selection_priority(strategy.inner_strategy, NodeView(root)))
    return PlungingContainer{typeof(root)}(nothing, open)
end

function select_node!(::PlungingNodeSelecStrategy, container::PlungingContainer)
    if container.dive !== nothing
        node = container.dive
        container.dive = nothing
        return node
    end
    return dequeue!(container.open)
end

function insert_node!(
    strategy::PlungingNodeSelecStrategy, container::PlungingContainer, node::SearchNode
)
    enqueue!(
        container.open,
        node,
        get_selection_priority(strategy.inner_strategy, NodeView(node)),
    )
    return nothing
end

"""
    insert_children!(::PlungingNodeSelecStrategy, container, children)

The first child becomes the dive node; the others go to the open nodes.
"""
function insert_children!(
    strategy::PlungingNodeSelecStrategy, container::PlungingContainer, children
)
    for child in children
        if container.dive === nothing
            container.dive = child
        else
            insert_node!(strategy, container, child)
        end
    end
    return nothing
end

function is_empty_container(container::PlungingContainer)
    return container.dive === nothing && isempty(container.open)
end
