# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    BranchingStatus

Status of a branching variable selection.
"""
@enum BranchingStatus branching_ok all_integral node_infeasible

"""
    BranchingDirection

Direction of a branching constraint: `branch_down` (≤ floor)
or `branch_up` (≥ ceil).
"""
@enum BranchingDirection branch_down branch_up

"""
    BranchingResult{X}

Result of a branching variable selection. Check `status` before
accessing `orig_var` or `value`.
"""
struct BranchingResult{X}
    status::BranchingStatus
    orig_var::Union{Nothing,X}
    value::Float64
end

BranchingResult(status::BranchingStatus) = BranchingResult{Nothing}(status, nothing, 0.0)

function BranchingResult(orig_var, value::Float64)
    return BranchingResult{typeof(orig_var)}(branching_ok, orig_var, value)
end

"""
    AbstractBranchingStrategy

Determines how to select a branching variable at each node.
"""
abstract type AbstractBranchingStrategy end

"""
    on_node_evaluated(strategy, space, node, cg_output)

Callback after CG completes on a node. Default: no-op.
"""
on_node_evaluated(::AbstractBranchingStrategy, space, node, cg_output) = nothing

"""
    MostFractionalBranching <: AbstractBranchingStrategy

Select the most fractional original variable for branching.
Delegates to `most_fractional_original_variable`.
"""
struct MostFractionalBranching <: AbstractBranchingStrategy end

"""
    select_branching_variable(strategy, space, node, primal_values)

Select a branching variable. Returns a `BranchingResult` with
`status` indicating the outcome.
"""
function select_branching_variable(
    ::MostFractionalBranching, space, node, primal_values::Dict{MOI.VariableIndex,Float64}
)
    orig_var, x_val = most_fractional_original_variable(
        space.ws, primal_values; tol=space.tolerance
    )
    if isnothing(orig_var)
        return BranchingResult(all_integral)
    end
    return BranchingResult(orig_var, x_val)
end

"""
    AbstractPriorityBranchingStrategy <: AbstractBranchingStrategy

Abstract type for branching strategies defined by a priority on each candidate. This is the
branching counterpart of `TreeSearch.AbstractBestFirstNodeSelecStrategy`: the strategy ranks the
fractional original variables and the one with the lowest priority is branched on.

Subtypes must implement [`get_branching_priority`](@ref). A strategy is an ordinary struct,
so it may hold parameters and state (a random number generator, say); the priority is
computed once per candidate and per node.

```julia
struct LargestValue <: Vertigo.Branching.AbstractPriorityBranchingStrategy end
function Vertigo.Branching.get_branching_priority(::LargestValue, c, node)
    return -c.value
end
```

Strategies that need more than a ranking (probing the candidates, as strong branching does)
implement [`select_branching_variable`](@ref) directly.
"""
abstract type AbstractPriorityBranchingStrategy <: AbstractBranchingStrategy end

"""
    get_branching_priority(strategy::AbstractPriorityBranchingStrategy,
        candidate::BranchingCandidate, node::NodeView) -> Float64

Return the priority of `candidate`, a fractional original variable of the master solution at
`node`, a read-only `TreeSearch.NodeView` (`BranchCutPrice` has accessors for its node data,
such as `parent_lp_obj`).

- **Lower is first**, as for node selection; use the opposite of a quantity to maximise it.
- **Ties** go to the candidate with the highest fractionality (the candidates come sorted
  by decreasing fractionality).
- **Candidate and node only.** The function does not see the search space or the master
  solution.
"""
function get_branching_priority end

function select_branching_variable(
    strategy::AbstractPriorityBranchingStrategy,
    space,
    node,
    primal_values::Dict{MOI.VariableIndex,Float64},
)
    candidates = find_fractional_variables(space.ws, primal_values; tol=space.tolerance)
    if isempty(candidates)
        return BranchingResult(all_integral)
    end
    view = NodeView(node)
    best = first(candidates)
    best_priority = get_branching_priority(strategy, best, view)
    for candidate in Iterators.drop(candidates, 1)
        priority = get_branching_priority(strategy, candidate, view)
        if priority < best_priority
            best, best_priority = candidate, priority
        end
    end
    return BranchingResult(best.orig_var, best.value)
end
