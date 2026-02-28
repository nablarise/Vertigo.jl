# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    DomainChangeTrackerHelper

Maintains mappings between variable indices and their bound constraints.
This is used to efficiently apply and track changes to variable bounds in an optimization model.

# Fields
- `map_lb`: Maps variable indices to their lower bound constraints.
- `map_ub`: Maps variable indices to their upper bound constraints.
- `map_eq`: Maps variable indices to their equality constraints.
- `map_integer`: Maps variable indices to their integrality constraints.
- `map_binary`: Maps variable indices to their binary constraints.
- `original_integer_vars`: Variable indices of originally integer variables.
- `original_binary_vars`: Variable indices of originally binary variables.
"""
struct DomainChangeTrackerHelper
    map_lb::Dict{MOI.VariableIndex, MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{Float64}}}
    map_ub::Dict{MOI.VariableIndex, MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{Float64}}}
    map_eq::Dict{MOI.VariableIndex, MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}}}
    map_integer::Dict{MOI.VariableIndex, MOI.ConstraintIndex{MOI.VariableIndex, MOI.Integer}}
    map_binary::Dict{MOI.VariableIndex, MOI.ConstraintIndex{MOI.VariableIndex, MOI.ZeroOne}}

    # Store variable index of original integer and binary variables.
    # Required to branch on a relaxed model.
    original_integer_vars::Set{MOI.VariableIndex}
    original_binary_vars::Set{MOI.VariableIndex}
    function DomainChangeTrackerHelper()
        return new(
            Dict{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}}(),
            Dict{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}}}(),
            Dict{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}}}(),
            Dict{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex,MOI.Integer}}(),
            Dict{MOI.VariableIndex,MOI.ConstraintIndex{MOI.VariableIndex,MOI.ZeroOne}}(),
            Set{MOI.VariableIndex}(),
            Set{MOI.VariableIndex}()
        )
    end
end

"""
    _register_constraints!(helper, vi, ci)

Register a constraint on a single variable in the appropriate mapping in the helper.

# Arguments
- `helper`: The DomainChangeTrackerHelper to update.
- `vi`: Variable index.
- `ci`: Constraint index involving only the variable `vi`.
"""
_register_constraints!(helper, vi, ci) = nothing

function _register_constraints!(helper, vi::F, ci::MOI.ConstraintIndex{F,S}) where {F<:MOI.VariableIndex,S<:MOI.GreaterThan}
    helper.map_lb[vi] = ci
end

function _register_constraints!(helper, vi::F, ci::MOI.ConstraintIndex{F,S}) where {F<:MOI.VariableIndex,S<:MOI.LessThan}
    helper.map_ub[vi] = ci
end

function _register_constraints!(helper, vi::F, ci::MOI.ConstraintIndex{F,S}) where {F<:MOI.VariableIndex,S<:MOI.EqualTo}
    helper.map_eq[vi] = ci
end

function _register_constraints!(helper, vi::F, ci::MOI.ConstraintIndex{F,S}) where {F<:MOI.VariableIndex,S<:MOI.Integer}
    helper.map_integer[vi] = ci
    push!(helper.original_integer_vars, vi)
end

function _register_constraints!(helper, vi::F, ci::MOI.ConstraintIndex{F,S}) where {F<:MOI.VariableIndex,S<:MOI.ZeroOne}
    helper.map_binary[vi] = ci
    push!(helper.original_binary_vars, vi)
end

"""
    LowerBoundVarChange <: AbstractAtomicChange

Represents a change to the lower bound of a variable.

# Fields
- `var_id::MOI.VariableIndex`: The index of the variable whose lower bound is changing.
- `new_lb::Float64`: The new lower bound value.
"""
struct LowerBoundVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    new_lb::Float64
end

"""
    apply_change!(backend, change::LowerBoundVarChange, helper::DomainChangeTrackerHelper)

Apply a lower bound change to a variable in the optimization model.

Creates a new constraint if one doesn't exist, otherwise updates the existing one.
Asserts that the variable is not fixed (no equality constraint).
"""
function apply_change!(backend, change::LowerBoundVarChange, helper::DomainChangeTrackerHelper)
    @assert !haskey(helper.map_eq, change.var_id)
    ci = get(helper.map_lb, change.var_id, nothing)
    if isnothing(ci)
        new_ci = MOI.add_constraint(backend, change.var_id, MOI.GreaterThan(change.new_lb))
        helper.map_lb[change.var_id] = new_ci
        @debug "add constraint $(change.var_id) => $(change.new_lb)"
    else
        MOI.set(backend, MOI.ConstraintSet(), ci, MOI.GreaterThan(change.new_lb))
        @debug "set constraint to $(change.var_id) => $(change.new_lb)"
    end
    return
end

"""
    UpperBoundVarChange <: AbstractAtomicChange

Represents a change to the upper bound of a variable.

# Fields
- `var_id::MOI.VariableIndex`: The index of the variable whose upper bound is changing.
- `new_ub::Float64`: The new upper bound value.
"""
struct UpperBoundVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    new_ub::Float64
end

"""
    apply_change!(backend, change::UpperBoundVarChange, helper::DomainChangeTrackerHelper)

Apply an upper bound change to a variable in the optimization model.

Creates a new constraint if one doesn't exist, otherwise updates the existing one.
Asserts that the variable is not fixed (no equality constraint).
"""
function apply_change!(backend, change::UpperBoundVarChange, helper::DomainChangeTrackerHelper)
    @assert !haskey(helper.map_eq, change.var_id)
    ci = get(helper.map_ub, change.var_id, nothing)

    if isnothing(ci)
       new_ci = MOI.add_constraint(backend, change.var_id, MOI.LessThan(change.new_ub))
       helper.map_ub[change.var_id] = new_ci
       @debug "add constraint $(change.var_id) <= $(change.new_ub)"
    else
        MOI.set(backend, MOI.ConstraintSet(), ci, MOI.LessThan(change.new_ub))
        @debug "set constraint to $(change.var_id) <= $(change.new_ub)"
    end
    return
end

"""
    DomainChangeDiff <: AbstractMathOptStateDiff

Represents a collection of changes to variable domains (bounds).

# Fields
- `lower_bounds::Dict{ColId,LowerBoundVarChange}`: Lower bound changes indexed by column ID.
- `upper_bounds::Dict{ColId,UpperBoundVarChange}`: Upper bound changes indexed by column ID.
"""
struct DomainChangeDiff <: AbstractMathOptStateDiff
    lower_bounds::Dict{ColId,LowerBoundVarChange}
    upper_bounds::Dict{ColId,UpperBoundVarChange}
end

function DomainChangeDiff(lb_changes::Vector{LowerBoundVarChange}, ub_changes::Vector{UpperBoundVarChange})
    return DomainChangeDiff(
        Dict(change.var_id.value => change for change in lb_changes),
        Dict(change.var_id.value => change for change in ub_changes)
    )
end

"""
    DomainChangeDiff()

Create an empty domain change difference with no bound changes.
"""
DomainChangeDiff() = DomainChangeDiff(
    Dict{ColId,LowerBoundVarChange}(),
    Dict{ColId,UpperBoundVarChange}(),
)

"""
    merge_forward_change_diff(parent_forward_diff::DomainChangeDiff, local_forward_change::DomainChangeDiff)

Merge a local forward domain change difference into a parent forward domain change difference.
Local changes take precedence when there are conflicts.
"""
function merge_forward_change_diff(parent_forward_diff::DomainChangeDiff, local_forward_change::DomainChangeDiff)
    child_lb_changes = copy(parent_forward_diff.lower_bounds)
    child_ub_changes = copy(parent_forward_diff.upper_bounds)

    for (col_id, change) in local_forward_change.lower_bounds
        child_lb_changes[col_id] = change
    end
    for (col_id, change) in local_forward_change.upper_bounds
        child_ub_changes[col_id] = change
    end
    return DomainChangeDiff(child_lb_changes, child_ub_changes)
end

"""
    merge_backward_change_diff(parent_backward_diff::DomainChangeDiff, local_backward_change::DomainChangeDiff)

Merge a parent backward domain change difference into a local backward domain change difference.
Parent changes take precedence when there are conflicts.
"""
function merge_backward_change_diff(parent_backward_diff::DomainChangeDiff, local_backward_change::DomainChangeDiff)
    child_lb_changes = copy(local_backward_change.lower_bounds)
    child_ub_changes = copy(local_backward_change.upper_bounds)

    for (col_id, change) in parent_backward_diff.lower_bounds
        child_lb_changes[col_id] = change
    end
    for (col_id, change) in parent_backward_diff.upper_bounds
        child_ub_changes[col_id] = change
    end
    return DomainChangeDiff(child_lb_changes, child_ub_changes)
end

"""
    apply_change!(backend, diff::DomainChangeDiff, helper)

Apply all domain changes in a difference to an optimization model.
"""
function apply_change!(backend, diff::DomainChangeDiff, helper)
    for change in values(diff.lower_bounds)
        apply_change!(backend, change, helper)
    end
    for change in values(diff.upper_bounds)
        apply_change!(backend, change, helper)
    end
    return
end

"""
    DomainChangeTracker <: AbstractMathOptStateTracker

Tracker for changes to variable domains (bounds) in an optimization model.
"""
struct DomainChangeTracker <: AbstractMathOptStateTracker end

"""
    root_state(::DomainChangeTracker, backend)

Create the root state for domain change tracking with empty forward and backward differences.
"""
root_state(::DomainChangeTracker, backend) = ModelState(DomainChangeDiff(), DomainChangeDiff())

"""
    new_state(::DomainChangeTracker, backward::DomainChangeDiff, forward::DomainChangeDiff)

Create a new model state with the given backward and forward domain change differences.
"""
new_state(::DomainChangeTracker, backward::DomainChangeDiff, forward::DomainChangeDiff) = ModelState(backward, forward)

"""
    transform_model!(::DomainChangeTracker, backend)

Transform a model for domain change tracking by creating a helper that maps
variables to their bound constraints.
"""
function transform_model!(::DomainChangeTracker, backend)
    helper = DomainChangeTrackerHelper()
    for (F, S) in MOI.get(backend, MOI.ListOfConstraintTypesPresent())
        if F == MOI.VariableIndex
            for ci in MOI.get(backend, MOI.ListOfConstraintIndices{F,S}())
                vi = MOI.get(backend, MOI.ConstraintFunction(), ci)
                _register_constraints!(helper, vi, ci)
            end
        end
    end
    return helper
end
