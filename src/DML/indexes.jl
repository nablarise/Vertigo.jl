# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# The index tuples a JuMP container is addressed by, as the user wrote them in
# the declaration (`x[k in ["a", "b"], t in 3:5]` is addressed by ("a", 3), not
# by positions). `obj[idx...]` returns the element for every container JuMP can
# build: Array, DenseAxisArray, SparseAxisArray and Dict.
_index_tuples(obj::AbstractArray) = (Tuple(i) for i in CartesianIndices(obj))
_index_tuples(obj::JuMP.Containers.DenseAxisArray) = Iterators.product(axes(obj)...)
_index_tuples(obj::JuMP.Containers.SparseAxisArray) = keys(obj.data)
_index_tuples(obj::AbstractDict) = (k isa Tuple ? k : (k,) for k in keys(obj))

_is_container(obj) = obj isa Union{AbstractArray,AbstractDict}

# Whether `obj` is a JuMP object of kind `R` (`JuMP.AbstractVariableRef` or
# `JuMP.ConstraintRef`), alone or in a container.
function _holds(obj, ::Type{R}) where {R}
    obj isa R && return true
    _is_container(obj) || return false
    isempty(obj) && return false
    return first(values(obj)) isa R
end

# `(name, idx)` for every `R` in the model, `idx` being `()` for a scalar.
function _named_objects(model, ::Type{R}) where {R}
    return (
        (name, idx)
        for (name, obj) in JuMP.object_dictionary(model) if _holds(obj, R)
        for idx in (obj isa R ? ((),) : _index_tuples(obj))
    )
end

JuMP.Containers.default_container(::Vector{Tuple{Int64,Int64}}) = JuMP.Containers.SparseAxisArray
JuMP.Containers.default_container(::Vector{<:Tuple}) = JuMP.Containers.SparseAxisArray
