# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    ColumnData

Data required to add or restore a pricing column in the master LP.

# Fields
- `id::Int`: Unique column identifier.
- `obj_coeff::Float64`: Objective coefficient.
- `lower_bound::Float64`: Variable lower bound (`Inf` = unbounded).
- `upper_bound::Float64`: Variable upper bound (`Inf` = unbounded).
- `entries::Vector{Tuple{LinearConstraintIndex, Float64}}`: Constraint coefficients.
"""
struct ColumnData
    id::Int
    obj_coeff::Float64
    lower_bound::Float64
    upper_bound::Float64
    entries::Vector{Tuple{LinearConstraintIndex, Float64}}
end

"""
    AddColumnChange <: AbstractAtomicChange

Atomic change that adds a pricing column to the master LP.
"""
struct AddColumnChange <: AbstractAtomicChange
    column::ColumnData
end

"""
    RemoveColumnChange <: AbstractAtomicChange

Atomic change that removes a pricing column from the master LP.
"""
struct RemoveColumnChange <: AbstractAtomicChange
    column::ColumnData
end

"""
    ColumnChangeDiff <: AbstractMathOptStateDiff

A collection of column additions and removals representing the difference
between two B&B tree nodes.

# Fields
- `add_columns::Vector{AddColumnChange}`: Columns to add.
- `remove_columns::Vector{RemoveColumnChange}`: Columns to remove.
"""
struct ColumnChangeDiff <: AbstractMathOptStateDiff
    add_columns::Vector{AddColumnChange}
    remove_columns::Vector{RemoveColumnChange}
end

"""
    ColumnChangeDiff()

Create an empty column change diff with no additions or removals.
"""
ColumnChangeDiff() = ColumnChangeDiff(AddColumnChange[], RemoveColumnChange[])

"""
    ColumnTrackerHelper

Maintains a mapping from column IDs to their active `MOI.VariableIndex`.

# Fields
- `active_columns::Dict{Int, MOI.VariableIndex}`: Maps column ID to variable index.
"""
mutable struct ColumnTrackerHelper
    active_columns::Dict{Int, MOI.VariableIndex}
end

"""
    ColumnTrackerHelper()

Create a helper with no active columns.
"""
ColumnTrackerHelper() = ColumnTrackerHelper(Dict{Int, MOI.VariableIndex}())

"""
    ColumnTracker <: AbstractMathOptStateTracker

Tracker for pricing columns added and removed during B&B tree navigation.

Uses a monotonically increasing counter to assign unique IDs to columns.
"""
struct ColumnTracker <: AbstractMathOptStateTracker
    next_col_id::Base.RefValue{Int}
end

"""
    ColumnTracker()

Create a new `ColumnTracker` with the counter initialised to zero.
"""
ColumnTracker() = ColumnTracker(Ref(0))

"""
    next_id!(tracker::ColumnTracker) -> Int

Increment and return the next unique column ID.
"""
function next_id!(tracker::ColumnTracker)
    tracker.next_col_id[] += 1
    return tracker.next_col_id[]
end

"""
    apply_change!(backend, change::AddColumnChange, helper::ColumnTrackerHelper)

Add the column to the backend and register its variable index in the helper.

Adds bound constraints for finite bounds and sets the objective and constraint
coefficients via `MOI.modify`.
"""
function apply_change!(backend, change::AddColumnChange, helper::ColumnTrackerHelper)
    col = change.column
    vi = MOI.add_variable(backend)
    helper.active_columns[col.id] = vi
    isfinite(col.lower_bound) &&
        MOI.add_constraint(backend, vi, MOI.GreaterThan(col.lower_bound))
    isfinite(col.upper_bound) &&
        MOI.add_constraint(backend, vi, MOI.LessThan(col.upper_bound))
    MOI.modify(
        backend,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarCoefficientChange(vi, col.obj_coeff),
    )
    for (ci, coeff) in col.entries
        MOI.modify(backend, ci, MOI.ScalarCoefficientChange(vi, coeff))
    end
    return
end

"""
    apply_change!(backend, change::RemoveColumnChange, helper::ColumnTrackerHelper)

Delete the column from the backend and unregister it from the helper.

`MOI.delete` removes the variable along with all its bound constraints and
clears its coefficients in all affine constraints (MOI spec guarantee).
"""
function apply_change!(backend, change::RemoveColumnChange, helper::ColumnTrackerHelper)
    col = change.column
    vi = helper.active_columns[col.id]
    MOI.delete(backend, vi)
    delete!(helper.active_columns, col.id)
    return
end

"""
    apply_change!(backend, diff::ColumnChangeDiff, helper::ColumnTrackerHelper)

Apply a full diff: remove first, then add.

Removals precede additions to prevent constraint conflicts when re-entering a node.
"""
function apply_change!(backend, diff::ColumnChangeDiff, helper::ColumnTrackerHelper)
    for change in diff.remove_columns
        apply_change!(backend, change, helper)
    end
    for change in diff.add_columns
        apply_change!(backend, change, helper)
    end
    return
end

"""
    merge_forward_change_diff(parent::ColumnChangeDiff, local_diff::ColumnChangeDiff)

Merge two forward diffs: parent operations first, local appended.

Forward traversal accumulates columns from root downward, so the parent's
operations must appear before the child's.
"""
function merge_forward_change_diff(
    parent::ColumnChangeDiff, local_diff::ColumnChangeDiff
)
    return ColumnChangeDiff(
        vcat(parent.add_columns, local_diff.add_columns),
        vcat(parent.remove_columns, local_diff.remove_columns),
    )
end

"""
    merge_backward_change_diff(parent::ColumnChangeDiff, local_diff::ColumnChangeDiff)

Merge two backward diffs: local unwind first, parent unwind appended.

Backward traversal unwinds the child before unwinding the parent.
"""
function merge_backward_change_diff(
    parent::ColumnChangeDiff, local_diff::ColumnChangeDiff
)
    return ColumnChangeDiff(
        vcat(local_diff.add_columns, parent.add_columns),
        vcat(local_diff.remove_columns, parent.remove_columns),
    )
end

"""
    root_state(::ColumnTracker, backend) -> ModelState{ColumnChangeDiff}

Return the root state with empty forward and backward diffs (no columns at root).
"""
root_state(::ColumnTracker, backend) =
    ModelState(ColumnChangeDiff(), ColumnChangeDiff())

"""
    new_state(::ColumnTracker, fwd, bwd) -> ModelState{ColumnChangeDiff}

Wrap forward and backward diffs into a `ModelState`.
"""
new_state(::ColumnTracker, fwd::ColumnChangeDiff, bwd::ColumnChangeDiff) =
    ModelState(fwd, bwd)

"""
    transform_model!(::ColumnTracker, backend) -> ColumnTrackerHelper

Initialise and return a `ColumnTrackerHelper` with no active columns.

No model transformation is needed — columns are managed entirely through
`apply_change!` during tree navigation.
"""
transform_model!(::ColumnTracker, backend) = ColumnTrackerHelper()
