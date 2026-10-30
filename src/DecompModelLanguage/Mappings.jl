# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    VariableMapping

Maps each variable of the original JuMP model to its copy in the master or in
a subproblem model, while those models are built.
"""
struct VariableMapping
    mapping::Dict{JuMP.VariableRef,JuMP.VariableRef}
end

VariableMapping() = VariableMapping(Dict{JuMP.VariableRef,JuMP.VariableRef}())

Base.getindex(m::VariableMapping, k::JuMP.VariableRef) = m.mapping[k]
function Base.setindex!(m::VariableMapping, v::JuMP.VariableRef, k::JuMP.VariableRef)
    return m.mapping[k] = v
end
Base.haskey(m::VariableMapping, k::JuMP.VariableRef) = haskey(m.mapping, k)
Base.iterate(m::VariableMapping) = iterate(m.mapping)
Base.iterate(m::VariableMapping, s) = iterate(m.mapping, s)

"""
    ConstraintMapping

Maps each constraint of the original JuMP model to its copy in the master or
in a subproblem model, while those models are built.
"""
struct ConstraintMapping
    mapping::Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}
end

ConstraintMapping() = ConstraintMapping(Dict{JuMP.ConstraintRef,JuMP.ConstraintRef}())

Base.getindex(m::ConstraintMapping, k::JuMP.ConstraintRef) = m.mapping[k]
function Base.setindex!(m::ConstraintMapping, v::JuMP.ConstraintRef, k::JuMP.ConstraintRef)
    return m.mapping[k] = v
end
Base.iterate(m::ConstraintMapping) = iterate(m.mapping)
Base.iterate(m::ConstraintMapping, s) = iterate(m.mapping, s)
