# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    LocalCut{S<:MOI.AbstractScalarSet}

A local cut to be added or removed from the optimization model during tree search.

# Fields
- `id::Int`: Unique identifier for the cut.
- `terms::Vector{MOI.ScalarAffineTerm{Float64}}`: The affine terms of the cut.
- `set::S`: The set defining the sense and RHS (e.g., `MOI.GreaterThan(rhs)`).
"""
struct LocalCut{S<:MOI.AbstractScalarSet}
    id::Int
    terms::Vector{MOI.ScalarAffineTerm{Float64}}
    set::S
end

"""
    AddLocalCutChange <: AbstractAtomicChange

Atomic change that adds a local cut to the optimization model.
"""
struct AddLocalCutChange <: AbstractAtomicChange
    cut::LocalCut
end

"""
    RemoveLocalCutChange <: AbstractAtomicChange

Atomic change that removes a local cut from the optimization model.
"""
struct RemoveLocalCutChange <: AbstractAtomicChange
    cut::LocalCut
end

"""
    LocalCutChangeDiff <: AbstractMathOptStateDiff

A collection of local cut additions and removals representing the difference
between two tree nodes.

# Fields
- `add_cuts::Vector{AddLocalCutChange}`: Cuts to add.
- `remove_cuts::Vector{RemoveLocalCutChange}`: Cuts to remove.
"""
struct LocalCutChangeDiff <: AbstractMathOptStateDiff
    add_cuts::Vector{AddLocalCutChange}
    remove_cuts::Vector{RemoveLocalCutChange}
end

"""
    LocalCutChangeDiff()

Create an empty local cut change diff with no additions or removals.
"""
LocalCutChangeDiff() = LocalCutChangeDiff(
    Vector{AddLocalCutChange}(),
    Vector{RemoveLocalCutChange}(),
)

"""
    LocalCutTrackerHelper

Maintains a mapping from cut IDs to their active constraint indices.

# Fields
- `active_cuts::Dict{Int,TaggedCI}`: Maps cut ID to its constraint index.
"""
mutable struct LocalCutTrackerHelper
    active_cuts::Dict{Int,TaggedCI}
end

"""
    LocalCutTrackerHelper()

Create a helper with no active cuts.
"""
LocalCutTrackerHelper() = LocalCutTrackerHelper(Dict{Int,TaggedCI}())

"""
    LocalCutTracker <: AbstractMathOptStateTracker

Tracker for locally-scoped cuts added and removed during B&B tree navigation.

Uses a monotonically increasing counter to assign unique IDs to cuts.
"""
struct LocalCutTracker <: AbstractMathOptStateTracker
    next_cut_id::Base.RefValue{Int}
end

"""
    LocalCutTracker()

Create a new `LocalCutTracker` with the counter initialised to zero.
"""
LocalCutTracker() = LocalCutTracker(Ref(0))

"""
    next_id!(tracker::LocalCutTracker) -> Int

Increment and return the next unique cut ID.
"""
function next_id!(tracker::LocalCutTracker)
    tracker.next_cut_id[] += 1
    return tracker.next_cut_id[]
end

"""
    apply_change!(backend, change::AddLocalCutChange, helper::LocalCutTrackerHelper)

Add the cut to the backend and register its constraint index in the helper.
"""
function apply_change!(
    backend, change::AddLocalCutChange, helper::LocalCutTrackerHelper
)
    cut = change.cut
    f = MOI.ScalarAffineFunction(cut.terms, 0.0)
    ci = MOI.add_constraint(backend, f, cut.set)
    helper.active_cuts[cut.id] = TaggedCI(ci)
    return
end

"""
    apply_change!(backend, change::RemoveLocalCutChange, helper::LocalCutTrackerHelper)

Delete the cut from the backend and unregister it from the helper.
"""
function apply_change!(
    backend, change::RemoveLocalCutChange, helper::LocalCutTrackerHelper
)
    cut = change.cut
    tagged = helper.active_cuts[cut.id]
    with_typed_ci(tagged) do ci
        MOI.delete(backend, ci)
    end
    delete!(helper.active_cuts, cut.id)
    return
end

"""
    apply_change!(backend, diff::LocalCutChangeDiff, helper::LocalCutTrackerHelper)

Apply a full diff: remove first, then add.

Removals precede additions to avoid constraint conflicts when re-entering a node.
"""
function apply_change!(
    backend, diff::LocalCutChangeDiff, helper::LocalCutTrackerHelper
)
    for change in diff.remove_cuts
        apply_change!(backend, change, helper)
    end
    for change in diff.add_cuts
        apply_change!(backend, change, helper)
    end
    return
end

"""
    merge_forward_change_diff(parent::LocalCutChangeDiff, local_diff::LocalCutChangeDiff)

Merge two forward diffs: parent cuts first, local cuts appended.

Forward traversal accumulates cuts from root downward, so the parent's operations
must appear before the child's.
"""
function merge_forward_change_diff(
    parent::LocalCutChangeDiff, local_diff::LocalCutChangeDiff
)
    return LocalCutChangeDiff(
        vcat(parent.add_cuts, local_diff.add_cuts),
        vcat(parent.remove_cuts, local_diff.remove_cuts),
    )
end

"""
    merge_backward_change_diff(parent::LocalCutChangeDiff, local_diff::LocalCutChangeDiff)

Merge two backward diffs: local unwind first, parent unwind appended.

Backward traversal unwinds the child before unwinding the parent.
"""
function merge_backward_change_diff(
    parent::LocalCutChangeDiff, local_diff::LocalCutChangeDiff
)
    return LocalCutChangeDiff(
        vcat(local_diff.add_cuts, parent.add_cuts),
        vcat(local_diff.remove_cuts, parent.remove_cuts),
    )
end

"""
    root_state(tracker::LocalCutTracker, backend)

Create the root state with empty forward and backward diffs (no cuts at root).
"""
function root_state(tracker::LocalCutTracker, backend)
    empty = LocalCutChangeDiff()
    return ModelState(empty, empty)
end

"""
    new_state(tracker::LocalCutTracker, forward_diff, backward_diff)

Wrap forward and backward diffs into a `ModelState`.
"""
function new_state(tracker::LocalCutTracker, forward_diff, backward_diff)
    return ModelState(forward_diff, backward_diff)
end

"""
    transform_model!(tracker::LocalCutTracker, backend)

Initialise and return a `LocalCutTrackerHelper` with no active cuts.

No model transformation is needed for local cuts — they are managed entirely
through `apply_change!` during tree navigation.
"""
function transform_model!(tracker::LocalCutTracker, backend)
    return LocalCutTrackerHelper()
end
