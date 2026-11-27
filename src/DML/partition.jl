# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# Author: Luis Marques <luismarques@luismarques.fr>
# SPDX-License-Identifier: MIT

# Scalars keep a `Set{Tuple{}}` of indices: the model builder dispatches on it.
function _push_index!(by_name, name, idx)
    indexes = get!(() -> idx == () ? Set{Tuple{}}() : Set{Tuple}(), by_name, name)
    push!(indexes, idx)
    return
end

# What the annotation assigns `R`s (variables or constraints) to: `name => indexes`
# for the master, and `sp_id => (name => indexes)` for the subproblems.
function _master_objects(model, dw_annotation, ::Type{R}) where {R}
    master = Dict{Symbol,Any}()
    for (name, idx) in _named_objects(model, R)
        if dw_annotation(Val(name), idx...) isa MasterAnnotation
            _push_index!(master, name, idx)
        end
    end
    return master
end

function _subproblem_objects(model, dw_annotation, ::Type{R}) where {R}
    partition = Dict{Any,Dict{Symbol,Any}}()
    for (name, idx) in _named_objects(model, R)
        annotation = dw_annotation(Val(name), idx...)
        if annotation isa SetupAnnotation && R === JuMP.ConstraintRef
            error(
                "Constraint $(_object_label(name, idx)) is annotated " *
                "setup(): only a variable can be a setup variable."
            )
        end
        if annotation isa Union{SubproblemAnnotation,SetupAnnotation}
            _push_index!(get!(Dict{Symbol,Any}, partition, annotation.id), name, idx)
        end
    end
    return partition
end

# The variable each subproblem declares with setup(): `sp_id => variable`.
function _setup_variables(model, dw_annotation)
    setup_vars = Dict{Any,JuMP.VariableRef}()
    for (name, idx) in _named_objects(model, JuMP.AbstractVariableRef)
        annotation = dw_annotation(Val(name), idx...)
        if annotation isa SetupAnnotation
            var = get_scalar_object(model, name, idx)
            if haskey(setup_vars, annotation.id)
                error(
                    "Subproblem $(annotation.id) has two setup variables, " *
                    "$(setup_vars[annotation.id]) and $var."
                )
            end
            _check_setup_domain(var)
            setup_vars[annotation.id] = var
        end
    end
    return setup_vars
end

# The dual bound counts the copies of a subproblem as the columns whose setup
# variable is 1: it must be binary, or fixed to 1.
function _check_setup_domain(var)
    is_fixed_to_one = JuMP.is_fixed(var) && JuMP.fix_value(var) == 1
    if !(JuMP.is_binary(var) || is_fixed_to_one)
        error("Setup variable $var must be binary or fixed to 1.")
    end
    return
end

_master_variables(model, ann) =
    _master_objects(model, ann, JuMP.AbstractVariableRef)
_partition_subproblem_variables(model, ann) =
    _subproblem_objects(model, ann, JuMP.AbstractVariableRef)
_master_constraints(model, ann) =
    _master_objects(model, ann, JuMP.ConstraintRef)
_partition_subproblem_constraints(model, ann) =
    _subproblem_objects(model, ann, JuMP.ConstraintRef)

# A constraint the annotation assigns to no model would be silently left out
# of the decomposition. An anonymous constraint is never even submitted to the
# annotation: it is not in the object dictionary.
function _check_all_constraints_assigned(model, master_constrs, sp_constrs_partition)
    assigned = Set{JuMP.ConstraintRef}()
    for by_name in Iterators.flatten(((master_constrs,), values(sp_constrs_partition)))
        for (name, indexes) in by_name, idx in indexes
            push!(assigned, get_scalar_object(model, name, idx))
        end
    end
    for constr in JuMP.all_constraints(model; include_variable_in_set_constraints=false)
        if !(constr in assigned)
            error(
                "Constraint $constr is assigned to neither the master nor a " *
                "subproblem. Annotate it; if it is anonymous, name it first: " *
                "only named constraints can be annotated."
            )
        end
    end
    return
end
