# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    _original_var_info(original_var) -> JuMP.VariableInfo

Read the bounds, the fixed value, the start value and the integrality (binary or integer) of
`original_var`, so that a copy of the variable can be built with the same properties.
"""
function _original_var_info(original_var)
    has_ub = JuMP.has_upper_bound(original_var)
    has_lb = JuMP.has_lower_bound(original_var)
    is_fixed = JuMP.is_fixed(original_var)
    return JuMP.VariableInfo(
        has_lb,
        has_lb ? JuMP.lower_bound(original_var) : -Inf,
        has_ub,
        has_ub ? JuMP.upper_bound(original_var) : Inf,
        is_fixed,
        is_fixed ? JuMP.fix_value(original_var) : nothing,
        JuMP.has_start_value(original_var),
        JuMP.start_value(original_var),
        JuMP.is_binary(original_var),
        JuMP.is_integer(original_var),
    )
end

"""
    _replace_vars_in_func(func, target_model, var_mapping; preserve_constant = true)

Rewrite the function `func` of the original model over the variables of `target_model`. Each
variable of `func` is replaced by its copy in `var_mapping`. A term on a variable that has no
copy in `target_model` is dropped: the variable belongs to another model, or to a
user-overridden subproblem, which has no model. This is why
[`check_ownerships`](@ref) rejects a subproblem constraint on a foreign variable beforehand.

`func` is an `AffExpr` or a single `JuMP.VariableRef`. With `preserve_constant = false`, the
constant of an `AffExpr` is left out, so that it is counted once, in the master. A single
variable that has no copy in `target_model` gives `0.0`.
"""
function _replace_vars_in_func(
    func::JuMP.AffExpr,
    target_model,
    var_mapping::VariableMapping;
    preserve_constant::Bool=true,
)
    # Variables of a user-overridden subproblem have no model, hence no mapping.
    terms = [
        var_mapping[var] => coeff for (var, coeff) in func.terms if
        haskey(var_mapping, var) && JuMP.owner_model(var_mapping[var]) == target_model
    ]
    constant = preserve_constant ? func.constant : 0.0
    return JuMP.AffExpr(constant, terms)
end

function _replace_vars_in_func(
    single_var::JuMP.VariableRef,
    target_model,
    var_mapping::VariableMapping;
    preserve_constant::Bool=true,
)
    if haskey(var_mapping, single_var)
        mapped = var_mapping[single_var]
        if JuMP.owner_model(mapped) == target_model
            return mapped
        end
    end
    return 0.0
end

"""
    _register_variables!(reform_model, var_mapping, original_vars)

Add to `reform_model` a copy of each variable of `original_vars`, with the same bounds, fixed
value, start value and integrality, and record it in `var_mapping`. The copies keep the name
of the originals.
"""
function _register_variables!(reform_model, var_mapping::VariableMapping, original_vars)
    for original_var in original_vars
        var = JuMP.build_variable(() -> error("todo."), _original_var_info(original_var))
        name =
            JuMP.set_string_names_on_creation(reform_model) ? JuMP.name(original_var) : ""
        var_mapping[original_var] = JuMP.add_variable(reform_model, var, name)
    end
    return nothing
end

"""
    _register_constraints!(reform_model, constr_mapping, original_constrs, var_mapping)

Add to `reform_model` a copy of each constraint of `original_constrs`, with the same set and
the function rewritten over the copies of `var_mapping` (see `_replace_vars_in_func`), and
record it in `constr_mapping`. The variables of the constraints have to be registered first.
The copies keep the name of the originals.
"""
function _register_constraints!(
    reform_model,
    constr_mapping::ConstraintMapping,
    original_constrs,
    var_mapping::VariableMapping,
)
    for original_constr in original_constrs
        original_constr_obj = JuMP.constraint_object(original_constr)
        mapped_func = _replace_vars_in_func(
            original_constr_obj.func, reform_model, var_mapping
        )
        constr = JuMP.build_constraint(
            () -> error("todo."), mapped_func, original_constr_obj.set
        )
        name = if JuMP.set_string_names_on_creation(reform_model)
            JuMP.name(original_constr)
        else
            ""
        end
        constr_mapping[original_constr] = JuMP.add_constraint(reform_model, constr, name)
    end
    return nothing
end

"""
    _register_objective!(reform_model, model, var_mapping; is_master = false)

Give `reform_model` the objective of `model`: the same sense, and the objective function
rewritten over the copies of `var_mapping`, so only the terms on the variables of
`reform_model` are kept. The constant of the objective is kept in the master (`is_master =
true`) only.
"""
function _register_objective!(
    reform_model, model, var_mapping::VariableMapping; is_master::Bool=false
)
    original_obj_func = JuMP.objective_function(model)
    original_obj_sense = JuMP.objective_sense(model)
    reform_obj_func = _replace_vars_in_func(
        original_obj_func, reform_model, var_mapping; preserve_constant=is_master
    )
    JuMP.set_objective_sense(reform_model, original_obj_sense)
    return JuMP.set_objective_function(reform_model, reform_obj_func)
end
