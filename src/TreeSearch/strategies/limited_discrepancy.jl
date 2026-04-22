# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    LimitedDiscrepancyStrategy{S<:AbstractSearchStrategy} <: AbstractSearchStrategy

Limited Discrepancy Search (LDS): wraps an inner search strategy and limits
the number of "non-preferred" branching decisions (discrepancies).

The discrepancy budget is tracked via the `discrepancy_remaining` field in the
user data. The user must include this field in their user data type.

# Convention
When `branch!` returns children, the FIRST child is the "preferred" direction
(discrepancy cost = 0) and subsequent children cost 1 discrepancy each.
Children that would exceed the remaining budget are filtered out.
"""
struct LimitedDiscrepancyStrategy{S<:AbstractSearchStrategy} <: AbstractSearchStrategy
    inner_strategy::S
    max_discrepancy::Int
end

# Delegate container operations to inner strategy
init_container(s::LimitedDiscrepancyStrategy, root::SearchNode) =
    init_container(s.inner_strategy, root)

select_node!(s::LimitedDiscrepancyStrategy, container) =
    select_node!(s.inner_strategy, container)

insert_node!(s::LimitedDiscrepancyStrategy, container, node::SearchNode) =
    insert_node!(s.inner_strategy, container, node)

"""
    DiscrepancyData{U}

Wrapper for user data that adds a discrepancy budget.
Used with `LimitedDiscrepancyStrategy` to track remaining discrepancies.
"""
struct DiscrepancyData{U}
    remaining_discrepancy::Int
    inner::U
end

"""
    filter_by_discrepancy(children, parent_discrepancy) -> Vector{SearchNode}

Filter children based on the discrepancy budget.
The first child (index 1) is the preferred direction and costs 0.
Each subsequent child (index i) costs (i-1) discrepancies.
Children whose cost exceeds the parent's remaining budget are dropped.

# Note
This function should be called by the user's `branch!` implementation when
using LDS. The framework does not call it automatically, because the user
controls child creation and must set the `DiscrepancyData` in user_data.
"""
function filter_by_discrepancy(children, parent_remaining::Int)
    filtered = typeof(first(children))[]
    for (i, child) in enumerate(children)
        cost = i - 1  # first child = preferred, cost 0
        if cost <= parent_remaining
            push!(filtered, child)
        end
    end
    return filtered
end
