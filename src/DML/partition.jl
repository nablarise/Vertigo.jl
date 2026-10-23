# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

_label(name, idx) = isempty(idx) ? string(name) : string(name, "[", join(idx, ", "), "]")

# Asks the annotation function which model owns `name[idx...]`. A missing rule
# surfaces as a MethodError on the annotation function itself; any other
# MethodError comes from inside a user's own function and is rethrown.
function _annotation(dw_annotation, kind::Symbol, name::Symbol, idx::Tuple)
    annotation = try
        dw_annotation(Val(name), idx...)
    catch err
        if err isa MethodError && err.f === dw_annotation &&
           !isempty(err.args) && err.args[1] === Val(name)
            throw(ArgumentError(
                "no rule assigns $kind `$(_label(name, idx))` to the master or " *
                "to a subproblem"
            ))
        end
        rethrow()
    end
    annotation isa Union{MasterAnnotation,SubproblemAnnotation} || throw(ArgumentError(
        "the rule for $kind `$(_label(name, idx))` returned $(repr(annotation)); " *
        "expected a master or subproblem annotation"
    ))
    return annotation
end

# Every variable and constraint has to be reachable by name: the rules address
# objects by name, so an anonymous one would silently stay out of the
# decomposition. Array-like containers (dense, `DenseAxisArray`,
# `SparseAxisArray`) are fine; `container=Dict` is not.
function _check_all_objects_named(model)
    named_vars = Set{JuMP.VariableRef}()
    named_constrs = Set{JuMP.ConstraintRef}()
    for (_, obj) in JuMP.object_dictionary(model)
        if obj isa JuMP.AbstractVariableRef
            push!(named_vars, obj)
        elseif obj isa JuMP.ConstraintRef
            push!(named_constrs, obj)
        elseif obj isa AbstractArray && length(obj) > 0
            if first(obj) isa JuMP.AbstractVariableRef
                union!(named_vars, obj)
            elseif first(obj) isa JuMP.ConstraintRef
                union!(named_constrs, obj)
            end
        end
    end
    for var in JuMP.all_variables(model)
        var in named_vars || throw(ArgumentError(
            "variable `$var` has no name the rules can address: declare it with a " *
            "name, e.g. `@variable(model, y[1:3])`, and not with `container=Dict`"
        ))
    end
    for constr in JuMP.all_constraints(model; include_variable_in_set_constraints=false)
        constr in named_constrs || throw(ArgumentError(
            "constraint `$constr` has no name the rules can address: declare it " *
            "with a name, e.g. `@constraint(model, c[i in I], ...)`, and not with " *
            "`container=Dict`"
        ))
    end
    return nothing
end

_function_variables(f::JuMP.AffExpr) = keys(f.terms)
_function_variables(f::JuMP.VariableRef) = (f,)
_function_variables(_) = ()

# A subproblem constraint is built over its own subproblem's variables only;
# a term on any other variable would be dropped, changing the model.
function _check_no_foreign_variables(
    model, master_vars, sp_vars_partition, sp_constrs_partition
)
    owner = Dict{JuMP.VariableRef,Any}()
    for (name, indexes) in master_vars, idx in indexes
        owner[get_scalar_object(model, name, idx)] = :master
    end
    for (sp_id, by_name) in sp_vars_partition, (name, indexes) in by_name, idx in indexes
        owner[get_scalar_object(model, name, idx)] = (:subproblem, sp_id)
    end
    for (sp_id, by_name) in sp_constrs_partition, (name, indexes) in by_name, idx in indexes
        constr = get_scalar_object(model, name, idx)
        for var in _function_variables(JuMP.constraint_object(constr).func)
            owner[var] == (:subproblem, sp_id) && continue
            where_ = owner[var] == :master ? "the master" : "subproblem $(owner[var][2])"
            throw(ArgumentError(
                "constraint `$(_label(name, idx))` belongs to subproblem $sp_id but " *
                "uses variable `$var` of $where_; a subproblem constraint can only " *
                "use the variables of its own subproblem"
            ))
        end
    end
    return nothing
end

function _master_variables(model, dw_annotation)
    master_vars = Dict{Symbol,Any}()
    for (var_name, var_obj) in JuMP.object_dictionary(model)
        if var_obj isa AbstractArray && length(var_obj) > 0 && first(var_obj) isa JuMP.AbstractVariableRef
            for idx in _eachindex(var_obj)
                annotation = _annotation(dw_annotation, :variable, var_name, Tuple(idx))
                if annotation isa MasterAnnotation
                    if !haskey(master_vars, var_name)
                        master_vars[var_name] = Set{Tuple}()
                    end
                    push!(master_vars[var_name], Tuple(idx))
                end
            end
        elseif var_obj isa JuMP.AbstractVariableRef
            annotation = _annotation(dw_annotation, :variable, var_name, ())
            if annotation isa MasterAnnotation
                if !haskey(master_vars, var_name)
                    master_vars[var_name] = Set{Tuple{}}([()])
                end
            end
        end
    end
    return master_vars
end

function _partition_subproblem_variables(model, dw_annotation)
    sp_vars = Dict{Any,Dict{Symbol,Any}}()
    for (var_name, var_obj) in JuMP.object_dictionary(model)
        if var_obj isa AbstractArray && length(var_obj) > 0 && first(var_obj) isa JuMP.AbstractVariableRef
            for idx in _eachindex(var_obj)
                annotation = _annotation(dw_annotation, :variable, var_name, Tuple(idx))
                if annotation isa SubproblemAnnotation
                    sp_dict = get!(Dict{Symbol,Any}, sp_vars, annotation.id)
                    idx_set = get!(Set{Tuple}, sp_dict, var_name)
                    push!(idx_set, Tuple(idx))
                end
            end
        elseif var_obj isa JuMP.AbstractVariableRef
            annotation = _annotation(dw_annotation, :variable, var_name, ())
            if annotation isa SubproblemAnnotation
                sp_dict = get!(Dict{Symbol,Any}, sp_vars, annotation.id)
                if !haskey(sp_dict, var_name)
                    sp_dict[var_name] = Set{Tuple{}}([()])
                end
            end
        end
    end
    return sp_vars
end

function _master_constraints(model, dw_annotation)
    master_constrs = Dict{Symbol,Any}()
    for (constr_name, constr_obj) in JuMP.object_dictionary(model)
        if constr_obj isa AbstractArray && length(constr_obj) > 0 && first(constr_obj) isa JuMP.ConstraintRef
            for idx in _eachindex(constr_obj)
                annotation = _annotation(dw_annotation, :constraint, constr_name, Tuple(idx))
                if annotation isa MasterAnnotation
                    if !haskey(master_constrs, constr_name)
                        master_constrs[constr_name] = Set{Tuple}()
                    end
                    push!(master_constrs[constr_name], Tuple(idx))
                end
            end
        elseif constr_obj isa JuMP.ConstraintRef
            annotation = _annotation(dw_annotation, :constraint, constr_name, ())
            if annotation isa MasterAnnotation
                if !haskey(master_constrs, constr_name)
                    master_constrs[constr_name] = Set{Tuple{}}([()])
                end
            end
        end
    end
    return master_constrs
end

function _partition_subproblem_constraints(model, dw_annotation)
    sp_constrs = Dict{Any,Dict{Symbol,Any}}()
    for (constr_name, constr_obj) in JuMP.object_dictionary(model)
        if constr_obj isa AbstractArray && length(constr_obj) > 0 && first(constr_obj) isa JuMP.ConstraintRef
            for idx in _eachindex(constr_obj)
                annotation = _annotation(dw_annotation, :constraint, constr_name, Tuple(idx))
                if annotation isa SubproblemAnnotation
                    sp_dict = get!(Dict{Symbol,Any}, sp_constrs, annotation.id)
                    idx_set = get!(Set{Tuple}, sp_dict, constr_name)
                    push!(idx_set, Tuple(idx))
                end
            end
        elseif constr_obj isa JuMP.ConstraintRef
            annotation = _annotation(dw_annotation, :constraint, constr_name, ())
            if annotation isa SubproblemAnnotation
                sp_dict = get!(Dict{Symbol,Any}, sp_constrs, annotation.id)
                if !haskey(sp_dict, constr_name)
                    sp_dict[constr_name] = Set{Tuple{}}([()])
                end
            end
        end
    end
    return sp_constrs
end
