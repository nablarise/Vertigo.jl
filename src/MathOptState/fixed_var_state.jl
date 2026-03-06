# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    FixVarChange <: AbstractAtomicChange

Represents a change that fixes a variable to a specific value.

# Fields
- `var_id::MOI.VariableIndex`: The index of the variable to fix.
- `value::Float64`: The value to fix the variable to.
"""
struct FixVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    value::Float64
end

"""
    apply_change!(backend, change::FixVarChange, helper::DomainChangeTrackerHelper)

Apply a fix variable change to an optimization model.

Asserts that the variable is not already fixed. Removes any existing lower and upper
bound constraints, then adds a new equality constraint fixing the variable.
"""
function apply_change!(backend, change::FixVarChange, helper::DomainChangeTrackerHelper)
    haskey(helper.map_eq, change.var_id) && error(
        "variable $(change.var_id) is already fixed"
    )
    if haskey(helper.map_lb, change.var_id)
        MOI.delete(backend, helper.map_lb[change.var_id])
        delete!(helper.map_lb, change.var_id)
    end
    if haskey(helper.map_ub, change.var_id)
        MOI.delete(backend, helper.map_ub[change.var_id])
        delete!(helper.map_ub, change.var_id)
    end
    eq_ci = MOI.add_constraint(backend, change.var_id, MOI.EqualTo(change.value))
    helper.map_eq[change.var_id] = eq_ci
    return
end

"""
    UnfixVarChange <: AbstractAtomicChange

Represents a change that unfixes a variable and resets bounds.

# Fields
- `var_id::MOI.VariableIndex`: The index of the variable to unfix.
- `lower_bound::Float64`: The initial lower bound of the variable.
- `upper_bound::Float64`: The initial upper bound of the variable.
"""
struct UnfixVarChange <: AbstractAtomicChange
    var_id::MOI.VariableIndex
    lower_bound::Float64
    upper_bound::Float64
end

"""
    apply_change!(backend, change::UnfixVarChange, helper::DomainChangeTrackerHelper)

Apply an unfix variable change to an optimization model.

Only takes action if the variable is currently fixed. Removes the equality constraint
and restores initial lower and upper bound constraints.
"""
function apply_change!(backend, change::UnfixVarChange, helper::DomainChangeTrackerHelper)
    if haskey(helper.map_eq, change.var_id)
        MOI.delete(backend, helper.map_eq[change.var_id])
        delete!(helper.map_eq, change.var_id)
        lb_ci = MOI.add_constraint(backend, change.var_id, MOI.GreaterThan(change.lower_bound))
        ub_ci = MOI.add_constraint(backend, change.var_id, MOI.LessThan(change.upper_bound))
        helper.map_lb[change.var_id] = lb_ci
        helper.map_ub[change.var_id] = ub_ci
    end
    return
end

"""
    FixVarChangeDiff <: AbstractMathOptStateDiff

Represents a collection of changes to variable fixations.

# Fields
- `fix_vars::Dict{ColId,FixVarChange}`: Fix variable changes indexed by column ID.
- `unfix_vars::Dict{ColId,UnfixVarChange}`: Unfix variable changes indexed by column ID.
"""
struct FixVarChangeDiff <: AbstractMathOptStateDiff
    fix_vars::Dict{ColId,FixVarChange}
    unfix_vars::Dict{ColId,UnfixVarChange}
end

"""
    FixVarChangeDiff()

Create an empty fix variable change difference with no changes.
"""
FixVarChangeDiff() = FixVarChangeDiff(Dict{ColId,FixVarChange}(), Dict{ColId,UnfixVarChange}())

"""
    FixVarChangeDiff(fix_var_changes, unfix_var_changes)

Create a fix variable change difference from vectors of fix and unfix changes.
"""
function FixVarChangeDiff(fix_var_changes::Vector{FixVarChange}, unfix_var_changes::Vector{UnfixVarChange})
    fix_vars = Dict{ColId,FixVarChange}(change.var_id.value => change for change in fix_var_changes)
    unfix_vars = Dict{ColId,UnfixVarChange}(change.var_id.value => change for change in unfix_var_changes)
    return FixVarChangeDiff(fix_vars, unfix_vars)
end

"""
    merge_forward_change_diff(parent_forward_diff::FixVarChangeDiff, local_forward_change::FixVarChangeDiff)

Merge a local forward fix variable change difference into a parent forward fix variable change difference.
Local changes take precedence when there are conflicts.
"""
function merge_forward_change_diff(parent_forward_diff::FixVarChangeDiff, local_forward_change::FixVarChangeDiff)
    child_fix_vars = copy(parent_forward_diff.fix_vars)
    for (col_id, change) in local_forward_change.fix_vars
        child_fix_vars[col_id] = change
    end

    child_unfix_vars = copy(parent_forward_diff.unfix_vars)
    for (col_id, change) in local_forward_change.unfix_vars
        child_unfix_vars[col_id] = change
    end
    return FixVarChangeDiff(child_fix_vars, child_unfix_vars)
end

"""
    merge_backward_change_diff(parent_backward_diff::FixVarChangeDiff, local_backward_change::FixVarChangeDiff)

Merge a parent backward fix variable change difference into a local backward fix variable change difference.
Parent changes take precedence when there are conflicts.
"""
function merge_backward_change_diff(parent_backward_diff::FixVarChangeDiff, local_backward_change::FixVarChangeDiff)
    child_fix_vars = copy(local_backward_change.fix_vars)
    for (col_id, change) in parent_backward_diff.fix_vars
        child_fix_vars[col_id] = change
    end

    child_unfix_vars = copy(local_backward_change.unfix_vars)
    for (col_id, change) in parent_backward_diff.unfix_vars
        child_unfix_vars[col_id] = change
    end
    return FixVarChangeDiff(child_fix_vars, child_unfix_vars)
end

"""
    apply_change!(backend, diff::FixVarChangeDiff, helper::DomainChangeTrackerHelper)

Apply all fix variable changes in a difference to an optimization model.
"""
function apply_change!(backend, diff::FixVarChangeDiff, helper::DomainChangeTrackerHelper)
    for change in values(diff.fix_vars)
        apply_change!(backend, change, helper)
    end
    for change in values(diff.unfix_vars)
        apply_change!(backend, change, helper)
    end
    return
end

"""
    FixVarChangeTracker <: AbstractMathOptStateTracker

Tracker for changes to variable fixations in an optimization model.
"""
struct FixVarChangeTracker <: AbstractMathOptStateTracker end

"""
    root_state(::FixVarChangeTracker, backend)

Create the root state for fix variable change tracking with empty forward and backward differences.
"""
root_state(::FixVarChangeTracker, backend) = ModelState(FixVarChangeDiff(), FixVarChangeDiff())

"""
    new_state(::FixVarChangeTracker, backward::FixVarChangeDiff, forward::FixVarChangeDiff)

Create a new model state with the given backward and forward fix variable change differences.
"""
new_state(::FixVarChangeTracker, backward::FixVarChangeDiff, forward::FixVarChangeDiff) = ModelState(backward, forward)

"""
    transform_model!(::FixVarChangeTracker, backend)

Transform a model for fix variable change tracking by creating a helper that maps
variables to their bound constraints.
"""
function transform_model!(::FixVarChangeTracker, backend)
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
