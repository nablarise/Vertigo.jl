# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    CutRhsChange <: AbstractAtomicChange

Represents a change to the right-hand side of a cut constraint.

# Fields
- `constr_id`: The index of the cut constraint.
- `rhs::Float64`: The new right-hand side value.
"""
struct CutRhsChange <: AbstractAtomicChange
    constr_id::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}
    rhs::Float64
end

"""
    apply_change!(backend, change::CutRhsChange, helper)

Apply a cut RHS change to a constraint in the optimization model.
"""
function apply_change!(backend, change::CutRhsChange, helper)
    MOI.set(backend, MOI.ConstraintSet(), change.constr_id, MOI.GreaterThan(change.rhs))
    return
end

"""
    CutRhsChangeDiff <: AbstractMathOptStateDiff

Represents a collection of changes to cut constraint right-hand sides.

# Fields
- `cut_rhs::Dict{RowId,CutRhsChange}`: Cut RHS changes indexed by row ID.
"""
struct CutRhsChangeDiff <: AbstractMathOptStateDiff
    cut_rhs::Dict{RowId,CutRhsChange}
end

"""
    CutRhsChangeDiff()

Create an empty cut RHS change difference with no changes.
"""
CutRhsChangeDiff() = CutRhsChangeDiff(Dict{RowId,CutRhsChange}())

"""
    merge_forward_change_diff(parent_forward_diff::CutRhsChangeDiff, local_forward_change::CutRhsChangeDiff)

Merge a local forward cut RHS change difference into a parent forward cut RHS change difference.
Local changes take precedence when there are conflicts.
"""
function merge_forward_change_diff(parent_forward_diff::CutRhsChangeDiff, local_forward_change::CutRhsChangeDiff)
    child_cut_rhs_changes = copy(parent_forward_diff.cut_rhs)

    for (row_id, change) in local_forward_change.cut_rhs
        child_cut_rhs_changes[row_id] = change
    end
    return CutRhsChangeDiff(child_cut_rhs_changes)
end

"""
    merge_backward_change_diff(parent_backward_diff::CutRhsChangeDiff, local_backward_change::CutRhsChangeDiff)

Merge a parent backward cut RHS change difference into a local backward cut RHS change difference.
Parent changes take precedence when there are conflicts.
"""
function merge_backward_change_diff(parent_backward_diff::CutRhsChangeDiff, local_backward_change::CutRhsChangeDiff)
    child_cut_rhs_changes = copy(local_backward_change.cut_rhs)

    for (row_id, change) in parent_backward_diff.cut_rhs
        child_cut_rhs_changes[row_id] = change
    end
    return CutRhsChangeDiff(child_cut_rhs_changes)
end

"""
    apply_change!(backend, diff::CutRhsChangeDiff, helper)

Apply all cut RHS changes in a difference to an optimization model.
"""
function apply_change!(backend, diff::CutRhsChangeDiff, helper)
    for change in values(diff.cut_rhs)
        apply_change!(backend, change, helper)
    end
    return
end

"""
    CutsTracker <: AbstractMathOptStateTracker

Tracker for changes to cut constraint right-hand sides in an optimization model.
"""
struct CutsTracker <: AbstractMathOptStateTracker end

"""
    root_state(::CutsTracker, backend)

Create the root state for cut RHS change tracking with empty forward and backward differences.
"""
root_state(::CutsTracker, backend) = ModelState(CutRhsChangeDiff(), CutRhsChangeDiff())

"""
    new_state(::CutsTracker, backward::CutRhsChangeDiff, forward::CutRhsChangeDiff)

Create a new model state with the given backward and forward cut RHS change differences.
"""
new_state(::CutsTracker, backward::CutRhsChangeDiff, forward::CutRhsChangeDiff) = ModelState(backward, forward)
