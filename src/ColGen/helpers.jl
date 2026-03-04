# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

_inf(is_min) = is_min ? -Inf : Inf

"""
add_variable!(model; lower_bound, upper_bound, variable_type, constraint_coeffs, objective_coeff)

Add a new variable to a MOI model with specified bounds and constraint coefficients.

Keyword Arguments:
- lower_bound: Lower bound for the variable (default: nothing)
- upper_bound: Upper bound for the variable (default: nothing)
- variable_type: MOI constraint for variable type (e.g., MOI.Integer(), MOI.ZeroOne()) (default: nothing)
- constraint_coeffs: Dict mapping constraint references to coefficients (default: empty)
- objective_coeff: Objective coefficient for the new variable (default: 0.0)
- name: Name for the variable (default: nothing)

Returns:
- MOI.VariableIndex: Reference to the created variable
"""
function add_variable!(
    model;
    lower_bound = nothing,
    upper_bound = nothing,
    variable_type = nothing,
    constraint_coeffs::Dict{TaggedCI,Float64} = Dict{TaggedCI,Float64}(),
    objective_coeff::Float64 = 0.0,
    name = nothing
)
    var = MOI.add_variable(model)

    if !isnothing(name)
        MOI.set(model, MOI.VariableName(), var, name)
    end

    if !isnothing(lower_bound)
        MOI.add_constraint(model, var, MOI.GreaterThan(lower_bound))
    end
    if !isnothing(upper_bound)
        MOI.add_constraint(model, var, MOI.LessThan(upper_bound))
    end

    if !isnothing(variable_type)
        MOI.add_constraint(model, var, variable_type)
    end

    for (tagged_ci, coeff) in constraint_coeffs
        if coeff != 0.0
            with_typed_ci(tagged_ci) do ci
                MOI.modify(
                    model, ci,
                    MOI.ScalarCoefficientChange(var, coeff)
                )
            end
        end
    end

    if objective_coeff != 0.0
        MOI.modify(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarCoefficientChange(var, objective_coeff)
        )
    end

    return var
end

"""
add_constraint!(model, coeffs, constraint_set; name=nothing)

Add a new linear constraint to a MOI model.

Arguments:
- model: MOI model to modify
- coeffs: Dict mapping variable references to coefficients
- constraint_set: MOI constraint set instance (e.g., MOI.EqualTo(5.0), MOI.LessThan(10.0))
- name: Name for the constraint (default: nothing)

Returns:
- MOI.ConstraintIndex: Reference to the created constraint
"""
function add_constraint!(
    model,
    coeffs::Dict{MOI.VariableIndex,Float64},
    constraint_set::MOI.AbstractSet;
    name = nothing
)
    terms = [MOI.ScalarAffineTerm(coeff, var) for (var, coeff) in coeffs if coeff != 0.0]
    func = MOI.ScalarAffineFunction(terms, 0.0)
    constraint_ref = MOI.add_constraint(model, func, constraint_set)

    if !isnothing(name)
        MOI.set(model, MOI.ConstraintName(), constraint_ref, name)
    end

    return constraint_ref
end
