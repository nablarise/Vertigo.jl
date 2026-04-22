# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    NodeStatus

Result of evaluating a node in the search tree.

- `CUTOFF`: Node is pruned (infeasible, bounded, etc.). No children generated.
- `FEASIBLE`: A feasible solution was found at this node.
- `BRANCH`: Node must be branched into children.
"""
@enum NodeStatus begin
    CUTOFF
    FEASIBLE
    BRANCH
end
