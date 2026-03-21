# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    AbstractBranchingStrategy

Determines how to select a branching variable at each node.
"""
abstract type AbstractBranchingStrategy end

"""
    MostFractionalBranching <: AbstractBranchingStrategy

Select the most fractional original variable for branching.
Delegates to `most_fractional_original_variable`.
"""
struct MostFractionalBranching <: AbstractBranchingStrategy end

"""
    select_branching_variable(strategy, space, node, primal_values)

Select a variable to branch on. Returns `(orig_var, x_val)` or
`nothing` if all variables are integral. Tolerance comes from
`space.tol`.
"""
function select_branching_variable(
    ::MostFractionalBranching, space, node,
    primal_values::Dict{MOI.VariableIndex,Float64}
)
    orig_var, x_val = most_fractional_original_variable(
        space.ctx, primal_values; tol=space.tol
    )
    isnothing(orig_var) && return nothing
    return (orig_var, x_val)
end
