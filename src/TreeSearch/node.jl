# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    NodeIdCounter

Thread-safe counter for generating unique node IDs.
"""
mutable struct NodeIdCounter
    value::Int
    NodeIdCounter() = new(0)
end

"""
    next_id!(counter::NodeIdCounter) -> Int

Return the next unique node ID and increment the counter.
"""
function next_id!(counter::NodeIdCounter)
    counter.value += 1
    return counter.value
end

"""
    SearchNode{D,U}

A node in the search tree carrying local diffs for efficient state transitions.

# Fields
- `id::Int`: Unique identifier for debugging and logging.
- `parent`: Parent node (`nothing` for the root).
- `local_forward_diff::D`: Diff to apply when moving from parent to this node.
- `local_backward_diff::D`: Diff to apply when moving from this node back to parent.
- `depth::Int`: Depth in the tree (root = 0).
- `dual_bound::Float64`: Lower bound from the LP relaxation at this node.
- `is_active::Bool`: Whether this node is on the current active path.
- `user_data::U`: User-defined data (LP solution, branching info, etc.).
"""
mutable struct SearchNode{D,U}
    id::Int
    parent::Union{Nothing, SearchNode{D,U}}
    local_forward_diff::D
    local_backward_diff::D
    depth::Int
    dual_bound::Float64
    is_active::Bool
    user_data::U
end

"""
    root_node(id_counter, forward_diff, backward_diff, user_data; dual_bound=-Inf)

Create the root node of a search tree.

# Arguments
- `id_counter::NodeIdCounter`: Counter for generating unique IDs.
- `forward_diff::D`: Forward diff for the root (typically empty/identity).
- `backward_diff::D`: Backward diff for the root (typically empty/identity).
- `user_data::U`: User-defined data attached to the root.
- `dual_bound::Float64`: Initial dual bound (default: `-Inf`).
"""
function root_node(
    id_counter::NodeIdCounter,
    forward_diff::D,
    backward_diff::D,
    user_data::U;
    dual_bound::Float64 = -Inf
) where {D, U}
    return SearchNode{D,U}(
        next_id!(id_counter),
        nothing,            # no parent
        forward_diff,
        backward_diff,
        0,                  # depth = 0
        dual_bound,
        true,               # root is always active
        user_data
    )
end

"""
    child_node(id_counter, parent, local_forward_diff, local_backward_diff;
               dual_bound=parent.dual_bound, user_data=parent.user_data)

Create a child node of `parent`.

# Arguments
- `id_counter::NodeIdCounter`: Counter for generating unique IDs.
- `parent::SearchNode{D,U}`: The parent node.
- `local_forward_diff::D`: Diff to transition from parent to this child.
- `local_backward_diff::D`: Diff to transition from this child back to parent.
- `dual_bound::Float64`: Dual bound (default: inherited from parent).
- `user_data::U`: User data (default: inherited from parent).
"""
function child_node(
    id_counter::NodeIdCounter,
    parent::SearchNode{D,U},
    local_forward_diff::D,
    local_backward_diff::D;
    dual_bound::Float64 = parent.dual_bound,
    user_data::U = parent.user_data
) where {D,U}
    return SearchNode{D,U}(
        next_id!(id_counter),
        parent,
        local_forward_diff,
        local_backward_diff,
        parent.depth + 1,
        dual_bound,
        false,              # not active until transitioned to
        user_data
    )
end

"""
    is_root(node::SearchNode) -> Bool

Return `true` if `node` is the root of the tree.
"""
is_root(node::SearchNode) = isnothing(node.parent)

"""
    find_common_ancestor(current::SearchNode, target::SearchNode) -> SearchNode

Find the deepest common ancestor between `current` and `target`.

**Precondition**: All nodes on the active path from root to `current` have
`is_active == true`. This is maintained by `transition_to!`.

The algorithm walks up from `target` until it hits an active node, which must
be on the path from root to `current`, hence is a common ancestor.
"""
function find_common_ancestor(current::SearchNode, target::SearchNode)
    # Special case: same node
    current === target && return current

    node = target
    while node !== nothing && !node.is_active
        node = node.parent
    end

    # node should never be nothing if the tree is well-formed (root is always active)
    if isnothing(node)
        error("Could not find common ancestor. Is the root node active?")
    end

    return node
end

"""
    collect_path_from_ancestor(ancestor::SearchNode, target::SearchNode) -> Vector{SearchNode}

Collect the path from `ancestor` (exclusive) down to `target` (inclusive).
Returns nodes in top-down order (ancestor's child first, target last).
"""
function collect_path_from_ancestor(ancestor::SearchNode{D,U}, target::SearchNode{D,U}) where {D,U}
    path = SearchNode{D,U}[]
    node = target
    while node !== ancestor
        pushfirst!(path, node)
        node = node.parent
        if isnothing(node)
            error("Target is not a descendant of ancestor.")
        end
    end
    return path
end

"""
    transition_to!(current, target, apply_forward!, apply_backward!)

Transition the model state from `current` node to `target` node via their
common ancestor. This implements Algorithm 3.1 from Achterberg (2007).

# Arguments
- `current::SearchNode`: The node the model is currently at.
- `target::SearchNode`: The node to transition to.
- `apply_forward!(diff)`: Callback to apply a forward diff to the model.
- `apply_backward!(diff)`: Callback to apply a backward diff to the model.

# Details
1. Find the deepest common ancestor of `current` and `target`.
2. Walk up from `current` to ancestor, applying backward diffs and deactivating nodes.
3. Walk down from ancestor to `target`, applying forward diffs and activating nodes.

The `is_active` flags are updated so that the active path always reflects the
path from root to the current node.
"""
function transition_to!(
    current::SearchNode,
    target::SearchNode,
    apply_forward!::F,
    apply_backward!::B
) where {F, B}
    # Same node: nothing to do
    current === target && return target

    # Step 1: Find common ancestor
    ancestor = find_common_ancestor(current, target)

    # Step 2: Walk up from current to ancestor (apply backward diffs)
    node = current
    while node !== ancestor
        apply_backward!(node.local_backward_diff)
        node.is_active = false
        node = node.parent
    end

    # Step 3: Collect path from ancestor down to target
    path = collect_path_from_ancestor(ancestor, target)

    # Step 4: Walk down from ancestor to target (apply forward diffs)
    for path_node in path
        apply_forward!(path_node.local_forward_diff)
        path_node.is_active = true
    end

    return target
end
