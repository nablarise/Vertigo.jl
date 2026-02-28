# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    LinearConstraintSet

Union of scalar set types used in LP constraints that participate in basis tracking.
"""
const LinearConstraintSet = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
}

"""
    LinearConstraintIndex

Union of `MOI.ConstraintIndex` types for scalar affine constraints with `LinearConstraintSet` sets.
"""
const LinearConstraintIndex = Union{
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}},
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}},
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}},
}

"""
    LPBasisState

A snapshot of the LP basis: the status of all variables and linear constraints.

# Fields
- `var_status`: maps each variable index to its `MOI.BasisStatusCode`.
- `constr_status`: maps each SAF constraint index to its `MOI.BasisStatusCode`.
"""
struct LPBasisState
    var_status::Dict{MOI.VariableIndex, MOI.BasisStatusCode}
    constr_status::Dict{LinearConstraintIndex, MOI.BasisStatusCode}
end

"""
    LPBasisDiff <: AbstractMathOptStateDiff

Diff carrying an optional LP basis snapshot.

- Forward diff: holds the basis to restore when entering a node.
- Backward diff: always empty (`nothing` basis) — the next LP solve overwrites it.
"""
struct LPBasisDiff <: AbstractMathOptStateDiff
    basis::Union{LPBasisState, Nothing}
end

"""
    LPBasisDiff()

Create an empty diff with no basis (no-op when applied).
"""
LPBasisDiff() = LPBasisDiff(nothing)

"""
    apply_change!(backend, diff::LPBasisDiff, ::Nothing)

Restore the LP basis stored in `diff` by setting variable and constraint basis
statuses on `backend`. Silently skips any indices that are no longer valid
(e.g. constraints deleted after the snapshot was taken).

Basis setting is a best-effort warm-start: if the solver does not support
`MOI.VariableBasisStatus` or `MOI.ConstraintBasisStatus` as settable attributes,
the attempt is silently abandoned and the solver falls back to cold-starting.
Correctness is never affected — only the number of simplex iterations.

Does nothing if `diff.basis` is `nothing`.
"""
function apply_change!(backend, diff::LPBasisDiff, ::Nothing)
    isnothing(diff.basis) && return
    for (vi, status) in diff.basis.var_status
        MOI.is_valid(backend, vi) || continue
        try
            MOI.set(backend, MOI.VariableBasisStatus(), vi, status)
        catch
            # Solver does not expose basis warm-starting via standard MOI —
            # abort silently; the next solve will cold-start instead.
            return
        end
    end
    for (ci, status) in diff.basis.constr_status
        MOI.is_valid(backend, ci) || continue
        try
            MOI.set(backend, MOI.ConstraintBasisStatus(), ci, status)
        catch
            return
        end
    end
    return
end

"""
    merge_forward_change_diff(::LPBasisDiff, node_diff::LPBasisDiff) -> LPBasisDiff

Each node owns a full basis snapshot, so the child diff replaces the parent's —
there is no accumulation.
"""
merge_forward_change_diff(::LPBasisDiff, node_diff::LPBasisDiff) = node_diff

"""
    merge_backward_change_diff(::LPBasisDiff, ::LPBasisDiff) -> LPBasisDiff

Backward diff is always a no-op: the next LP solve overwrites the basis, so
there is nothing to undo.
"""
merge_backward_change_diff(::LPBasisDiff, ::LPBasisDiff) = LPBasisDiff()

"""
    capture_basis(backend) -> LPBasisState

Read and return the current LP basis from `backend`.

Captures the `VariableBasisStatus` for every variable and the
`ConstraintBasisStatus` for every `ScalarAffineFunction` constraint with a
`GreaterThan`, `LessThan`, or `EqualTo` set.
"""
function capture_basis(backend)
    var_status = Dict{MOI.VariableIndex, MOI.BasisStatusCode}()
    for vi in MOI.get(backend, MOI.ListOfVariableIndices())
        var_status[vi] = MOI.get(backend, MOI.VariableBasisStatus(), vi)
    end
    constr_status = Dict{LinearConstraintIndex, MOI.BasisStatusCode}()
    F = MOI.ScalarAffineFunction{Float64}
    for S in (MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.EqualTo{Float64})
        for ci in MOI.get(backend, MOI.ListOfConstraintIndices{F, S}())
            constr_status[ci] = MOI.get(backend, MOI.ConstraintBasisStatus(), ci)
        end
    end
    return LPBasisState(var_status, constr_status)
end

"""
    update_basis(state::ModelState{LPBasisDiff}, backend) -> ModelState{LPBasisDiff}

Return a new `ModelState` whose forward diff carries the current basis from
`backend`. The backward diff is carried over unchanged from `state`.
"""
function update_basis(state::ModelState{LPBasisDiff}, backend)
    return ModelState(LPBasisDiff(capture_basis(backend)), state.backward_diff)
end

"""
    LPBasisTracker <: AbstractMathOptStateTracker

Tracker that seeds the LP simplex with a node's captured basis before each solve,
enabling warm starts (Achterberg §3.3.6, step 10).

This tracker is asymmetric: the forward diff restores a basis snapshot; the
backward diff is always a no-op since the next LP solve overwrites the basis.
"""
struct LPBasisTracker <: AbstractMathOptStateTracker end

"""
    root_state(::LPBasisTracker, backend) -> ModelState{LPBasisDiff}

Return the root state with empty (no-op) forward and backward diffs.
"""
root_state(::LPBasisTracker, backend) = ModelState(LPBasisDiff(), LPBasisDiff())

"""
    new_state(::LPBasisTracker, forward_diff, backward_diff) -> ModelState{LPBasisDiff}

Wrap forward and backward diffs into a `ModelState`.
"""
new_state(::LPBasisTracker, forward_diff::LPBasisDiff, backward_diff::LPBasisDiff) =
    ModelState(forward_diff, backward_diff)

"""
    transform_model!(::LPBasisTracker, backend)

No model transformation is needed for basis warm-starting — return `nothing`.
"""
transform_model!(::LPBasisTracker, backend) = nothing
