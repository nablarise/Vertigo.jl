# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# Author: Luis Marques <luismarques@luismarques.fr>
# SPDX-License-Identifier: MIT

# The index tuples a JuMP container is addressed by, as the user wrote them in
# the declaration (`x[k in ["a", "b"], t in 3:5]` is addressed by ("a", 3), not
# by positions). `obj[idx...]` returns the element for every container JuMP can
# build: Array, DenseAxisArray, SparseAxisArray and Dict.
_index_tuples(obj::AbstractArray) = (Tuple(i) for i in CartesianIndices(obj))
_index_tuples(obj::JuMP.Containers.DenseAxisArray) = Iterators.product(axes(obj)...)
_index_tuples(obj::JuMP.Containers.SparseAxisArray) = keys(obj.data)
_index_tuples(obj::AbstractDict) = (k isa Tuple ? k : (k,) for k in keys(obj))

# Whether `obj` is a JuMP object of kind `R` (`JuMP.AbstractVariableRef` or
# `JuMP.ConstraintRef`), alone or in a non-empty container.
_holds(obj, ::Type{R}) where {R} = obj isa R
function _holds(obj::Union{AbstractArray,AbstractDict}, ::Type{R}) where {R}
    return !isempty(obj) && first(values(obj)) isa R
end

# `(name, idx)` for every `R` in the model, `idx` being `()` for a scalar.
function _named_objects(model, ::Type{R}) where {R}
    return (
        (name, idx)
        for (name, obj) in JuMP.object_dictionary(model) if _holds(obj, R)
        for idx in (obj isa R ? ((),) : _index_tuples(obj))
    )
end
