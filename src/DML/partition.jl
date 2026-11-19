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
        if annotation isa SubproblemAnnotation
            _push_index!(get!(Dict{Symbol,Any}, partition, annotation.id), name, idx)
        end
    end
    return partition
end

_master_variables(model, ann) =
    _master_objects(model, ann, JuMP.AbstractVariableRef)
_partition_subproblem_variables(model, ann) =
    _subproblem_objects(model, ann, JuMP.AbstractVariableRef)
_master_constraints(model, ann) =
    _master_objects(model, ann, JuMP.ConstraintRef)
_partition_subproblem_constraints(model, ann) =
    _subproblem_objects(model, ann, JuMP.ConstraintRef)
