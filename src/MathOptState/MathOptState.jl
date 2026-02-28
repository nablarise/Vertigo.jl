# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module MathOptState

using MathOptInterface

const MOI = MathOptInterface
const ColId = Int
const RowId = Int

"""
    ModelState{T}

Encapsulates the changes required to restore the model state when navigating a search tree
(e.g., in a branch-and-bound algorithm), where each node corresponds to a specific model state.

- `backward_diff`: modifications to undo in order to revert the model to the root formulation.
- `forward_diff`: modifications to reapply in order to reconstruct the formulation corresponding
  to the current `ModelState` object.

This structure enables efficient model state management by avoiding full reconstruction
when switching between nodes in the search tree.
"""
struct ModelState{T}
    forward_diff::T
    backward_diff::T
end

"""
    backward(model_state::ModelState)

Get the backward difference from a model state, which contains the modifications
needed to undo in order to revert the model to the root formulation.
"""
backward(model_state::ModelState) = model_state.backward_diff

"""
    forward(model_state::ModelState)

Get the forward difference from a model state, which contains the modifications
needed to reapply in order to reconstruct the formulation corresponding to the
current `model_state`.
"""
forward(model_state::ModelState) = model_state.forward_diff

"""
    recover_state!(backend, prev_state::ModelState, next_state::ModelState, helper)

Transition a model from one `prev_state` to another `next_state` in the search tree by
applying the backward difference of the previous state (to revert to root) and then the
forward difference of the next state (to advance to the target state).

# Arguments
- `backend`: The MOI model to modify.
- `prev_state::ModelState`: The previous state.
- `next_state::ModelState`: The next state.
- `helper`: Helper object that assists with applying changes to the model.
"""
function recover_state!(backend, prev_state::ModelState, next_state::ModelState, helper)
    apply_change!(backend, backward(prev_state), helper)
    apply_change!(backend, forward(next_state), helper)
end

"""
    AbstractAtomicChange

Abstract type representing an atomic change to a mathematical optimization model.
Subtypes should implement specific types of changes (e.g., variable bounds, cuts, etc.).
"""
abstract type AbstractAtomicChange end

"""
    apply_change!(backend, change::AbstractAtomicChange, helper)

Apply an atomic change to a mathematical optimization model.

# Arguments
- `backend`: The MOI model to modify.
- `change::AbstractAtomicChange`: The atomic change to apply.
- `helper`: Helper object that assists with applying changes to the model.

# Returns
Nothing.
"""
function apply_change! end

"""
    AbstractMathOptStateTracker

Abstract type representing a tracker for mathematical optimization model states.
Subtypes should implement specific state tracking strategies for different aspects
of the optimization model (e.g., variable bounds, cuts, fixed variables).

This type is used to implement the state tracking interface.
"""
abstract type AbstractMathOptStateTracker end

"""
    AbstractMathOptStateDiff

Abstract type representing a difference between two states of a mathematical optimization model.
Subtypes should implement specific difference types for different aspects of the model state
(e.g., domain changes, cut changes, fixed variable changes).

This is used to track changes between model states and enable transitions between states.
"""
abstract type AbstractMathOptStateDiff end

"""
    merge_forward_change_diff(parent_forward_diff, local_forward_change)

Merge a local forward change difference into a parent forward change difference.

# Arguments
- `parent_forward_diff`: The parent forward difference.
- `local_forward_change`: The local forward difference to merge.

# Returns
A new diff that combines both differences.
"""
function merge_forward_change_diff end

"""
    merge_backward_change_diff(parent_backward_diff, local_backward_change)

Merge a local backward change difference into a parent backward change difference.

# Arguments
- `parent_backward_diff`: The parent backward difference.
- `local_backward_change`: The local backward difference to merge.

# Returns
A new diff that combines both differences.
"""
function merge_backward_change_diff end

"""
    root_state(math_opt_state_tracker, model)

Create the root state for a mathematical optimization model using a state tracker.

# Arguments
- `math_opt_state_tracker`: The state tracker to use.
- `model`: The optimization model.

# Returns
A `ModelState` representing the root state of the model.
"""
function root_state end

"""
    helper(math_opt_state_tracker, model)

Create a helper object for a mathematical optimization model using a state tracker.

# Arguments
- `math_opt_state_tracker`: The state tracker to use.
- `model`: The optimization model.

# Returns
A helper object that assists with applying changes to the model.
"""
function helper end

"""
    transform_model!(math_opt_state_tracker, model)

Transform a model to make state tracking easier.

This function allows for model transformations that facilitate state tracking.
It should not modify the original model's behavior, only its internal representation.

# Arguments
- `math_opt_state_tracker`: The state tracker to use.
- `model`: The optimization model to transform.

# Returns
A helper object that assists with applying changes to the transformed model.

# Note
This function should not change the original model's optimization behavior!
"""
function transform_model! end

"""
    new_state(math_opt_state_tracker, backward, forward)

Create a new model state with the given backward and forward differences.

# Arguments
- `math_opt_state_tracker`: The state tracker to use.
- `backward`: The backward difference for the new state.
- `forward`: The forward difference for the new state.

# Returns
A new `ModelState` with the specified differences.
"""
function new_state end


### Default Implementations
include("var_bounds_state.jl")
include("cut_rhs_state.jl")
include("fixed_var_state.jl")
include("integrality_state.jl")
include("local_cut_state.jl")
include("lp_basis_state.jl")

# Bridge to TreeSearch.transition_to! — must come after all trackers are defined
include("transition.jl")

export ModelState, backward, forward, recover_state!
export AbstractAtomicChange, apply_change!
export AbstractMathOptStateTracker, AbstractMathOptStateDiff
export merge_forward_change_diff, merge_backward_change_diff
export root_state, helper, transform_model!, new_state
export DomainChangeTrackerHelper, DomainChangeTracker
export LowerBoundVarChange, UpperBoundVarChange, DomainChangeDiff
export CutRhsChange, CutRhsChangeDiff, CutsTracker
export FixVarChange, UnfixVarChange, FixVarChangeDiff, FixVarChangeTracker
export IntegralityStateTracker, IntegralityChange, IntegralityChangeDiff
export IntegralityChangeType, relax_integrality, restrict_integrality
export relax_zero_one, restrict_zero_one, relax_integrality!
export LocalCut, AddLocalCutChange, RemoveLocalCutChange
export LocalCutChangeDiff, LocalCutTracker, LocalCutTrackerHelper, next_id!
export LinearConstraintSet, LinearConstraintIndex
export LPBasisState, LPBasisDiff, LPBasisTracker
export capture_basis, update_basis
export make_transition_callbacks

end # module MathOptState
