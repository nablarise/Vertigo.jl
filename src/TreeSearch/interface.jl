# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    AbstractSearchStrategy

Abstract type for search strategies (DFS, BFS, Best-First, etc.).
Subtypes control the order in which nodes are explored.
"""
abstract type AbstractSearchStrategy end

"""
    AbstractSearchSpace

Abstract type representing the problem-specific search space.
The user implements this to define the problem structure, model transitions,
and branching logic.

# Required methods
- `new_root(space) -> SearchNode`
- `stop(space, open_nodes) -> Bool`
- `output(space) -> Any`
- `transition!(space, current::SearchNode, next::SearchNode)`
- `branch!(space, node::SearchNode) -> iterable of SearchNode`

# Optional methods
- `on_feasible_solution!(space, node::SearchNode)` — default: no-op
- `prune!(space, open_nodes)` — default: no-op
"""
abstract type AbstractSearchSpace end

"""
    AbstractNodeEvaluator

Abstract type for node evaluation strategies.
The user implements this to define how a node is processed (LP solve,
propagation, heuristics, etc.).

# Required methods
- `evaluate!(evaluator, space, node::SearchNode) -> NodeStatus`
"""
abstract type AbstractNodeEvaluator end

# ============================================================
# Required interface — SearchSpace
# ============================================================

"""
    new_root(space::AbstractSearchSpace) -> SearchNode

Create and return the root node of the search tree.
The model should be in its root state when this is called.
"""
function new_root end

"""
    stop(space::AbstractSearchSpace, open_nodes) -> Bool

Return `true` if the search should terminate.
Called before each node selection. `open_nodes` is the strategy-specific container.
"""
function stop end

"""
    output(space::AbstractSearchSpace) -> Any

Extract and return the final result after the search completes.
"""
function output end

"""
    transition!(space::AbstractSearchSpace, current::SearchNode, next::SearchNode)

Apply the model state transition from `current` to `next`.
The implementation should call `transition_to!` from the node module,
providing callbacks that apply diffs to the underlying optimization model.

# Example implementation
```julia
function TreeSearch.transition!(space::MySpace, current, next)
    transition_to!(current, next,
        diff -> MathOptState.apply_change!(space.backend, diff, space.helper),
        diff -> MathOptState.apply_change!(space.backend, diff, space.helper)
    )
end
```
"""
function transition! end

"""
    branch!(space::AbstractSearchSpace, node::SearchNode) -> iterable of SearchNode

Generate child nodes for `node`. Called only when `evaluate!` returns `BRANCH`.
Children should be created with `child_node(...)`.
Return an iterable (Vector, tuple, generator) of `SearchNode`.
"""
function branch! end

# ============================================================
# Optional interface — SearchSpace
# ============================================================

"""
    on_feasible_solution!(space::AbstractSearchSpace, node::SearchNode)

Called when `evaluate!` returns `FEASIBLE`. Update the incumbent solution.
Default implementation does nothing.
"""
on_feasible_solution!(space::AbstractSearchSpace, node::SearchNode) = nothing

"""
    prune!(space::AbstractSearchSpace, open_nodes)

Prune nodes from the open set whose dual bound exceeds the incumbent.
Default implementation does nothing.
"""
prune!(space::AbstractSearchSpace, open_nodes) = nothing

# ============================================================
# Required interface — NodeEvaluator
# ============================================================

"""
    evaluate!(evaluator::AbstractNodeEvaluator, space::AbstractSearchSpace, node::SearchNode) -> NodeStatus

Evaluate the current node. The model is guaranteed to be in the state
corresponding to `node` when this is called.

Must return one of:
- `CUTOFF`: Node is pruned. No children will be generated.
- `FEASIBLE`: A feasible solution was found.
- `BRANCH`: Node should be branched.

The evaluator has full freedom in what it does: solve LP, propagate domains,
call heuristics, add cuts, etc.
"""
function evaluate! end

# ============================================================
# Strategy container interface
# ============================================================

"""
    init_container(strategy::AbstractSearchStrategy, root::SearchNode)

Create and return the initial open-node container for the given strategy,
with `root` already inserted.
"""
function init_container end

"""
    select_node!(strategy::AbstractSearchStrategy, container)

Select and remove the next node to process from `container`.
"""
function select_node! end

"""
    insert_node!(strategy::AbstractSearchStrategy, container, node::SearchNode)

Insert `node` into `container`.
"""
function insert_node! end

"""
    is_empty_container(container) -> Bool

Return `true` if the container has no nodes.
"""
function is_empty_container end

"""
    insert_children!(strategy, container, children)

Insert multiple child nodes into the container. The default implementation
calls `insert_node!` for each child. Strategies like DFS override this
to insert in reverse order (so the first child is explored first).
"""
function insert_children!(strategy::AbstractSearchStrategy, container, children)
    for child in children
        insert_node!(strategy, container, child)
    end
    return
end
