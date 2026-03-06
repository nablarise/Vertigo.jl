# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct VariableMapping
    mapping::Dict{JuMP.VariableRef,JuMP.VariableRef}
end

VariableMapping() = VariableMapping(Dict{JuMP.VariableRef,JuMP.VariableRef}())

Base.getindex(m::VariableMapping, k::JuMP.VariableRef) = m.mapping[k]
Base.setindex!(m::VariableMapping, v::JuMP.VariableRef, k::JuMP.VariableRef) = m.mapping[k] = v
Base.haskey(m::VariableMapping, k::JuMP.VariableRef) = haskey(m.mapping, k)
Base.iterate(m::VariableMapping) = iterate(m.mapping)
Base.iterate(m::VariableMapping, s) = iterate(m.mapping, s)

struct ConstraintMapping
    mapping::Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}
end

ConstraintMapping() = ConstraintMapping(Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}())

Base.getindex(m::ConstraintMapping, k::JuMP.ConstraintRef) = m.mapping[k]
Base.setindex!(m::ConstraintMapping, v::JuMP.ConstraintRef, k::JuMP.ConstraintRef) = m.mapping[k] = v
Base.iterate(m::ConstraintMapping) = iterate(m.mapping)
Base.iterate(m::ConstraintMapping, s) = iterate(m.mapping, s)

function get_scalar_object(model, object_name, index)
    return JuMP.object_dictionary(model)[object_name][index...]
end

function get_scalar_object(model, object_name, ::Tuple{})
    return JuMP.object_dictionary(model)[object_name]
end

function _original_var_info(original_model, var_name, index)
    original_var = get_scalar_object(original_model, var_name, index)
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

function _replace_vars_in_func(
    func::JuMP.AffExpr, target_model,
    var_mapping::VariableMapping;
    preserve_constant::Bool=true
)
    terms = [
        var_mapping[var] => coeff
        for (var, coeff) in func.terms
        if JuMP.owner_model(var_mapping[var]) == target_model
    ]
    constant = preserve_constant ? func.constant : 0.0
    return JuMP.AffExpr(constant, terms...)
end

function _replace_vars_in_func(
    single_var::JuMP.VariableRef, target_model,
    var_mapping::VariableMapping;
    preserve_constant::Bool=true
)
    mapped = var_mapping[single_var]
    if JuMP.owner_model(mapped) == target_model
        return mapped
    end
    return 0.0
end

function _register_variables!(
    reform_model, var_mapping::VariableMapping,
    original_model, var_infos_by_names
)
    for (var_name, var_infos_by_indexes) in var_infos_by_names
        _register_variable_object!(
            reform_model, var_mapping,
            original_model, var_name, var_infos_by_indexes
        )
    end
end

function _register_variable_object!(
    reform_model, var_mapping::VariableMapping,
    original_model, var_name,
    var_infos_by_indexes::Dict{Tuple{},<:Any}
)
    index = ()
    var_info = var_infos_by_indexes[index]
    var = JuMP.build_variable(() -> error("todo."), var_info)
    jump_var_name = if JuMP.set_string_names_on_creation(reform_model)
        string(var_name)
    else
        ""
    end
    original_var = get_scalar_object(original_model, var_name, index)
    reform_var = JuMP.add_variable(reform_model, var, jump_var_name)
    var_mapping[original_var] = reform_var
    reform_model[var_name] = reform_var
end

function _register_variable_object!(
    reform_model, var_mapping::VariableMapping,
    original_model, var_name, var_infos_by_indexes
)
    indexes = sort(collect(keys(var_infos_by_indexes)))
    vars = JuMP.Containers.container(
        (index...) -> begin
            var = JuMP.build_variable(
                () -> error("todo."),
                var_infos_by_indexes[index]
            )
            jump_var_name = if JuMP.set_string_names_on_creation(reform_model)
                JuMP.string(var_name, "[", JuMP.string(index), "]")
            else
                ""
            end
            original_var = get_scalar_object(
                original_model, var_name, index
            )
            var_mapping[original_var] = JuMP.add_variable(
                reform_model, var, jump_var_name
            )
        end,
        indexes
    )
    reform_model[var_name] = vars
end

function _register_constraints!(
    reform_model, constr_mapping::ConstraintMapping,
    original_model, constr_by_names,
    var_mapping::VariableMapping
)
    for (constr_name, constr_by_indexes) in constr_by_names
        _register_constraint_object!(
            reform_model, constr_mapping,
            original_model, constr_name,
            constr_by_indexes, var_mapping
        )
    end
end

function _register_constraint_object!(
    reform_model, constr_mapping::ConstraintMapping,
    original_model, constr_name,
    constr_by_indexes::Set{Tuple{}},
    var_mapping::VariableMapping
)
    index = ()
    original_constr = get_scalar_object(
        original_model, constr_name, index
    )
    original_constr_obj = JuMP.constraint_object(original_constr)
    mapped_func = _replace_vars_in_func(
        original_constr_obj.func,
        reform_model,
        var_mapping
    )
    constr = JuMP.build_constraint(
        () -> error("todo."), mapped_func, original_constr_obj.set
    )
    jump_constr_name = if JuMP.set_string_names_on_creation(reform_model)
        string(constr_name)
    else
        ""
    end
    reform_constr = JuMP.add_constraint(
        reform_model, constr, jump_constr_name
    )
    constr_mapping[original_constr] = reform_constr
    reform_model[constr_name] = reform_constr
end

function _register_constraint_object!(
    reform_model, constr_mapping::ConstraintMapping,
    original_model, constr_name, constr_by_indexes,
    var_mapping::VariableMapping
)
    indexes = sort(collect(constr_by_indexes))
    constrs = JuMP.Containers.container(
        (index...) -> begin
            original_constr = get_scalar_object(
                original_model, constr_name, index
            )
            original_constr_obj = JuMP.constraint_object(original_constr)
            mapped_func = _replace_vars_in_func(
                original_constr_obj.func,
                reform_model,
                var_mapping
            )
            constr = JuMP.build_constraint(
                () -> error("todo."),
                mapped_func,
                original_constr_obj.set
            )
            jump_constr_name = if JuMP.set_string_names_on_creation(reform_model)
                JuMP.string(constr_name, "[", JuMP.string(index), "]")
            else
                ""
            end
            constr_mapping[original_constr] = JuMP.add_constraint(
                reform_model, constr, jump_constr_name
            )
        end,
        indexes
    )
    reform_model[constr_name] = constrs
end

function _register_objective!(
    reform_model, model, var_mapping::VariableMapping;
    is_master::Bool=false
)
    original_obj_func = JuMP.objective_function(model)
    original_obj_sense = JuMP.objective_sense(model)
    reform_obj_func = _replace_vars_in_func(
        original_obj_func, reform_model, var_mapping;
        preserve_constant=is_master
    )
    JuMP.set_objective_sense(reform_model, original_obj_sense)
    JuMP.set_objective_function(reform_model, reform_obj_func)
end
