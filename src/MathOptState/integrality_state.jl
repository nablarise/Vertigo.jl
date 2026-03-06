# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct IntegralityStateTracker <: AbstractMathOptStateTracker end

function _apply_integrality_relaxation!(backend, var_id, helper)
    ci = get(helper.map_integer, var_id, nothing)
    if !isnothing(ci)
        MOI.delete(backend, ci)
        delete!(helper.map_integer, var_id)
    else
        error("cannot relax variable $var_id: not integral")
    end
    return
end

function _apply_integrality_restriction!(backend, var_id, helper)
    ci = get(helper.map_integer, var_id, nothing)
    if isnothing(ci)
        ci = MOI.add_constraint(backend, var_id, MOI.Integer())
        helper.map_integer[var_id] = ci
    else
        error("variable $var_id is already integral")
    end
    return
end

function _apply_zero_one_relaxation!(backend, var_id, helper)
    ci = get(helper.map_binary, var_id, nothing)
    if !isnothing(ci)
        MOI.delete(backend, ci)
        delete!(helper.map_binary, var_id)

        # Check bounds
        ci_lb = get(helper.map_lb, var_id, nothing)
        if isnothing(ci_lb) || MOI.get(backend, MOI.ConstraintSet(), ci_lb).lower < 0
            !isnothing(ci_lb) && MOI.delete(backend, ci_lb)
            new_ci_lb = MOI.add_constraint(backend, var_id, MOI.GreaterThan(0.0))
            helper.map_lb[var_id] = new_ci_lb
        end

        ci_ub = get(helper.map_ub, var_id, nothing)
        if isnothing(ci_ub) || MOI.get(backend, MOI.ConstraintSet(), ci_ub).upper > 1
            !isnothing(ci_ub) && MOI.delete(backend, ci_ub)
            new_ci_ub = MOI.add_constraint(backend, var_id, MOI.LessThan(1.0))
            helper.map_ub[var_id] = new_ci_ub
        end
    else
        error("cannot relax variable $var_id: not binary")
    end
    return
end

function _apply_zero_one_restriction!(backend, var_id, helper)
    ci = get(helper.map_binary, var_id, nothing)
    if isnothing(ci)
        ci = MOI.add_constraint(backend, var_id, MOI.ZeroOne())
        helper.map_binary[var_id] = ci
    else
        error("variable $var_id is already binary")
    end
    return
end

@enum IntegralityChangeType relax_integrality restrict_integrality relax_zero_one restrict_zero_one

struct IntegralityChange
    var_id::MOI.VariableIndex
    change::IntegralityChangeType
end

function apply_change!(backend, change::IntegralityChange, helper::DomainChangeTrackerHelper)
    if change.change == relax_integrality
        _apply_integrality_relaxation!(backend, change.var_id, helper)
    elseif change.change == restrict_integrality
        _apply_integrality_restriction!(backend, change.var_id, helper)
    elseif change.change == relax_zero_one
        _apply_zero_one_relaxation!(backend, change.var_id, helper)
    elseif change.change == restrict_zero_one
        _apply_zero_one_restriction!(backend, change.var_id, helper)
    end
end

struct IntegralityChangeDiff <: AbstractMathOptStateDiff
    integrality_changes::Vector{IntegralityChange}
end

IntegralityChangeDiff() = IntegralityChangeDiff(IntegralityChange[])

function apply_change!(backend, change::IntegralityChangeDiff, helper::DomainChangeTrackerHelper)
    for c in change.integrality_changes
        apply_change!(backend, c, helper)
    end
    return
end

new_state(::IntegralityStateTracker, backward::IntegralityChangeDiff, forward::IntegralityChangeDiff) = ModelState(backward, forward)

"""
    relax_integrality!(backend, helper::DomainChangeTrackerHelper)

Create a `ModelState` that relaxes all integrality constraints in the model.
Returns a state with forward diffs (relax) and backward diffs (restrict).
"""
function relax_integrality!(backend, helper::DomainChangeTrackerHelper)
    # Relax integrality constraints (Integer and ZeroOne) and add bounds for binary variables.
    forward_changes_itr = Iterators.flatten((
        Iterators.map(var_id -> IntegralityChange(var_id, relax_integrality), keys(helper.map_integer)),
        Iterators.map(var_id -> IntegralityChange(var_id, relax_zero_one), keys(helper.map_binary))
    ))
    forward_changes = IntegralityChangeDiff(collect(forward_changes_itr))

    backward_changes_itr = Iterators.flatten((
        Iterators.map(var_id -> IntegralityChange(var_id, restrict_integrality), keys(helper.map_integer)),
        Iterators.map(var_id -> IntegralityChange(var_id, restrict_zero_one), keys(helper.map_binary))
    ))
    backward_changes = IntegralityChangeDiff(collect(backward_changes_itr))

    return new_state(IntegralityStateTracker(), forward_changes, backward_changes)
end
