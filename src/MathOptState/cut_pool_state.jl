# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    GlobalCut

A globally-valid cut that can be activated or deactivated at each B&B node.
The set can be `GreaterThan`, `LessThan`, or `EqualTo`.

# Fields
- `id::Int`: Unique identifier.
- `terms::Vector{MOI.ScalarAffineTerm{Float64}}`: Affine terms.
- `set::LinearConstraintSet`: The constraint set (sense + RHS).
"""
struct GlobalCut
    id::Int
    terms::Vector{MOI.ScalarAffineTerm{Float64}}
    set::LinearConstraintSet
end

"""
    ActivateGlobalCutChange <: AbstractAtomicChange

Atomic change that adds a global cut to the LP (activates it).
"""
struct ActivateGlobalCutChange <: AbstractAtomicChange
    cut::GlobalCut
end

"""
    DeactivateGlobalCutChange <: AbstractAtomicChange

Atomic change that removes a global cut from the LP (deactivates it).
"""
struct DeactivateGlobalCutChange <: AbstractAtomicChange
    cut::GlobalCut
end

"""
    GlobalCutPoolHelper

Maintains a mapping from cut IDs to their active constraint indices.

Only cuts currently present in the LP are tracked here; deactivated cuts
are absent from the dict.

# Fields
- `active_cuts::Dict{Int, TaggedCI}`: Maps cut ID to its constraint index.
"""
mutable struct GlobalCutPoolHelper
    active_cuts::Dict{Int, TaggedCI}
end

"""
    GlobalCutPoolHelper()

Create a helper with no active cuts.
"""
GlobalCutPoolHelper() = GlobalCutPoolHelper(Dict{Int, TaggedCI}())

"""
    GlobalCutPoolDiff <: AbstractMathOptStateDiff

A keyed collection of cut activations and deactivations representing the
difference between two B&B tree nodes.

Dict-based merge semantics: a cut either gets activated or deactivated at a
given node — the last assignment wins (unlike `LocalCutChangeDiff` which uses
vcat accumulation).

# Fields
- `activate::Dict{Int, ActivateGlobalCutChange}`: Cuts to add (keyed by cut ID).
- `deactivate::Dict{Int, DeactivateGlobalCutChange}`: Cuts to remove (keyed by cut ID).
"""
struct GlobalCutPoolDiff <: AbstractMathOptStateDiff
    activate::Dict{Int, ActivateGlobalCutChange}
    deactivate::Dict{Int, DeactivateGlobalCutChange}
end

"""
    GlobalCutPoolDiff()

Create an empty diff with no activations or deactivations.
"""
GlobalCutPoolDiff() = GlobalCutPoolDiff(
    Dict{Int, ActivateGlobalCutChange}(),
    Dict{Int, DeactivateGlobalCutChange}(),
)

"""
    apply_change!(backend, change::ActivateGlobalCutChange, helper::GlobalCutPoolHelper)

Add the global cut to the backend if not already active.
"""
function apply_change!(
    backend, change::ActivateGlobalCutChange, helper::GlobalCutPoolHelper
)
    cut = change.cut
    haskey(helper.active_cuts, cut.id) && return
    ci = MOI.add_constraint(
        backend,
        MOI.ScalarAffineFunction(cut.terms, 0.0),
        cut.set,
    )
    helper.active_cuts[cut.id] = TaggedCI(ci)
    return
end

"""
    apply_change!(backend, change::DeactivateGlobalCutChange, helper::GlobalCutPoolHelper)

Remove the global cut from the backend if currently active.
"""
function apply_change!(
    backend, change::DeactivateGlobalCutChange, helper::GlobalCutPoolHelper
)
    cut = change.cut
    tagged = get(helper.active_cuts, cut.id, nothing)
    isnothing(tagged) && return
    with_typed_ci(tagged) do ci
        MOI.delete(backend, ci)
    end
    delete!(helper.active_cuts, cut.id)
    return
end

"""
    apply_change!(backend, diff::GlobalCutPoolDiff, helper::GlobalCutPoolHelper)

Apply a full diff: deactivate first, then activate.

Deactivations precede activations to avoid constraint conflicts when
re-entering a node.
"""
function apply_change!(
    backend, diff::GlobalCutPoolDiff, helper::GlobalCutPoolHelper
)
    for change in values(diff.deactivate)
        apply_change!(backend, change, helper)
    end
    for change in values(diff.activate)
        apply_change!(backend, change, helper)
    end
    return
end

"""
    merge_forward_change_diff(parent::GlobalCutPoolDiff, local_diff::GlobalCutPoolDiff)

Merge two forward diffs: parent entries first, local entries override.

Cross-cleaning ensures a cut cannot appear in both `activate` and `deactivate`:
if the parent activated a cut that the local diff deactivates, the activation
is removed, and vice versa.
"""
function merge_forward_change_diff(
    parent::GlobalCutPoolDiff, local_diff::GlobalCutPoolDiff
)
    activate = copy(parent.activate)
    deactivate = copy(parent.deactivate)
    for (id, change) in local_diff.activate
        activate[id] = change
        delete!(deactivate, id)
    end
    for (id, change) in local_diff.deactivate
        deactivate[id] = change
        delete!(activate, id)
    end
    return GlobalCutPoolDiff(activate, deactivate)
end

"""
    merge_backward_change_diff(parent::GlobalCutPoolDiff, local_diff::GlobalCutPoolDiff)

Merge two backward diffs: local entries first, parent entries override.

Backward traversal unwinds the child before unwinding the parent; cross-cleaning
maintains the invariant that no cut appears in both `activate` and `deactivate`.
"""
function merge_backward_change_diff(
    parent::GlobalCutPoolDiff, local_diff::GlobalCutPoolDiff
)
    activate = copy(local_diff.activate)
    deactivate = copy(local_diff.deactivate)
    for (id, change) in parent.activate
        activate[id] = change
        delete!(deactivate, id)
    end
    for (id, change) in parent.deactivate
        deactivate[id] = change
        delete!(activate, id)
    end
    return GlobalCutPoolDiff(activate, deactivate)
end

"""
    GlobalCutPoolTracker <: AbstractMathOptStateTracker

Tracker for globally-valid cuts that are selectively activated at each B&B node.

Maintains a pool of all known cuts and a counter for unique IDs. Use
`register_cut!` to add a cut to the pool, then build `GlobalCutPoolDiff`
values to activate/deactivate cuts per node.
"""
struct GlobalCutPoolTracker <: AbstractMathOptStateTracker
    next_cut_id::Base.RefValue{Int}
    pool::Dict{Int, GlobalCut}
end

"""
    GlobalCutPoolTracker()

Create a new `GlobalCutPoolTracker` with an empty pool.
"""
GlobalCutPoolTracker() = GlobalCutPoolTracker(Ref(0), Dict{Int, GlobalCut}())

"""
    register_cut!(tracker::GlobalCutPoolTracker, terms, set) -> GlobalCut

Register a new cut in the global pool and return the `GlobalCut` object.

The `set` must be a `GreaterThan`, `LessThan`, or `EqualTo` scalar set.
"""
function register_cut!(
    tracker::GlobalCutPoolTracker,
    terms::Vector{MOI.ScalarAffineTerm{Float64}},
    set::LinearConstraintSet,
)::GlobalCut
    tracker.next_cut_id[] += 1
    id = tracker.next_cut_id[]
    cut = GlobalCut(id, terms, set)
    tracker.pool[id] = cut
    return cut
end

"""
    root_state(::GlobalCutPoolTracker, backend) -> ModelState{GlobalCutPoolDiff}

Return the root state with empty forward and backward diffs (no cuts active at root).
"""
root_state(::GlobalCutPoolTracker, backend) =
    ModelState(GlobalCutPoolDiff(), GlobalCutPoolDiff())

"""
    new_state(::GlobalCutPoolTracker, fwd, bwd) -> ModelState{GlobalCutPoolDiff}

Wrap forward and backward diffs into a `ModelState`.
"""
new_state(
    ::GlobalCutPoolTracker,
    fwd::GlobalCutPoolDiff,
    bwd::GlobalCutPoolDiff,
) = ModelState(fwd, bwd)

"""
    transform_model!(::GlobalCutPoolTracker, backend) -> GlobalCutPoolHelper

Initialise and return a `GlobalCutPoolHelper` with no active cuts.
"""
transform_model!(::GlobalCutPoolTracker, backend) = GlobalCutPoolHelper()
