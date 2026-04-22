# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    search(strategy, space, evaluator)

Execute a tree search with the given strategy, search space, and node evaluator.

# The search loop

1. Create the root node via `new_root(space)`.
2. Initialize the open-node container via `init_container(strategy, root)`.
3. Loop until `stop(space, container)` or the container is empty:
   a. Select next node via `select_node!(strategy, container)`.
   b. Transition the model from current to next via `transition!(space, current, next)`.
   c. Evaluate the node via `evaluate!(evaluator, space, node)`.
   d. Depending on the status:
      - `BRANCH`: Generate children via `branch!(space, node)` and insert into container.
      - `FEASIBLE`: Call `on_feasible_solution!(space, node)` and `prune!(space, container)`.
      - `CUTOFF`: Do nothing (node is dead).
4. Return `output(space)`.

# Arguments
- `strategy::AbstractSearchStrategy`: Controls node selection order.
- `space::AbstractSearchSpace`: Problem-specific search space.
- `evaluator::AbstractNodeEvaluator`: Controls how nodes are evaluated.

# Returns
Whatever `output(space)` returns.
"""
function search(
    strategy::AbstractSearchStrategy,
    space::AbstractSearchSpace,
    evaluator::AbstractNodeEvaluator
)
    root = new_root(space)
    container = init_container(strategy, root)
    current_node = root
    is_first = true

    while !stop(space, container) && !is_empty_container(container)
        next_node = select_node!(strategy, container)

        # Transition model state (skip for the root — model is already there)
        if is_first
            is_first = false
        else
            transition!(space, current_node, next_node)
        end

        # Evaluate
        status = evaluate!(evaluator, space, next_node)

        if status == BRANCH
            insert_children!(strategy, container, branch!(space, next_node))
        elseif status == FEASIBLE
            on_feasible_solution!(space, next_node)
            prune!(space, container)
        end
        # CUTOFF: nothing to do

        current_node = next_node
    end

    return output(space)
end
