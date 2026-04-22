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

BranchingResult(status::BranchingStatus) =
    BranchingResult{Nothing}(status, nothing, 0.0)

function BranchingResult(orig_var, value::Float64)
    return BranchingResult{typeof(orig_var)}(
        branching_ok, orig_var, value
    )
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
    ::MostFractionalBranching, space, node,
    primal_values::Dict{MOI.VariableIndex,Float64}
)
    orig_var, x_val = most_fractional_original_variable(
        space.ws, primal_values; tol=space.tol
    )
    isnothing(orig_var) && return BranchingResult(all_integral)
    return BranchingResult(orig_var, x_val)
end
