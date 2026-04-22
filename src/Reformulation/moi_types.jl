# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    PricingSubproblemId

Concrete identifier for a pricing subproblem.
"""
struct PricingSubproblemId
    value::Int
end

const _SAF = MOI.ScalarAffineFunction{Float64}

const _VI = MOI.VariableIndex

@enum CIKind::UInt8 begin
    SAF_EQ   # ConstraintIndex{SAF, EqualTo{Float64}}
    SAF_LEQ  # ConstraintIndex{SAF, LessThan{Float64}}
    SAF_GEQ  # ConstraintIndex{SAF, GreaterThan{Float64}}
    VI_EQ    # ConstraintIndex{VariableIndex, EqualTo{Float64}}
    VI_LEQ   # ConstraintIndex{VariableIndex, LessThan{Float64}}
    VI_GEQ   # ConstraintIndex{VariableIndex, GreaterThan{Float64}}
end

function _ci_type(kind::CIKind)
    kind == SAF_EQ  && return MOI.ConstraintIndex{_SAF,MOI.EqualTo{Float64}}
    kind == SAF_LEQ && return MOI.ConstraintIndex{_SAF,MOI.LessThan{Float64}}
    kind == SAF_GEQ && return MOI.ConstraintIndex{_SAF,MOI.GreaterThan{Float64}}
    kind == VI_EQ   && return MOI.ConstraintIndex{_VI,MOI.EqualTo{Float64}}
    kind == VI_LEQ  && return MOI.ConstraintIndex{_VI,MOI.LessThan{Float64}}
    return MOI.ConstraintIndex{_VI,MOI.GreaterThan{Float64}}
end

"""
    TaggedCI

Concrete isbits representation of a `MOI.ConstraintIndex{SAF,S}`.
Stores the index value and a `CIKind` tag to recover the concrete
type at MOI call boundaries via `with_typed_ci`.
"""
struct TaggedCI
    value::Int64
    kind::CIKind
end
@assert isbitstype(TaggedCI)
function Base.isless(a::TaggedCI, b::TaggedCI)
    return (a.kind, a.value) < (b.kind, b.value)
end

TaggedCI(ci::MOI.ConstraintIndex{_SAF,MOI.EqualTo{Float64}}) =
    TaggedCI(ci.value, SAF_EQ)
TaggedCI(ci::MOI.ConstraintIndex{_SAF,MOI.LessThan{Float64}}) =
    TaggedCI(ci.value, SAF_LEQ)
TaggedCI(ci::MOI.ConstraintIndex{_SAF,MOI.GreaterThan{Float64}}) =
    TaggedCI(ci.value, SAF_GEQ)
TaggedCI(ci::MOI.ConstraintIndex{_VI,MOI.EqualTo{Float64}}) =
    TaggedCI(ci.value, VI_EQ)
TaggedCI(ci::MOI.ConstraintIndex{_VI,MOI.LessThan{Float64}}) =
    TaggedCI(ci.value, VI_LEQ)
TaggedCI(ci::MOI.ConstraintIndex{_VI,MOI.GreaterThan{Float64}}) =
    TaggedCI(ci.value, VI_GEQ)
TaggedCI(ci::MOI.ConstraintIndex{F,S}) where {F,S} =
    error("unsupported constraint type: $F-in-$S")

@inline function with_typed_ci(f, idx::TaggedCI)
    ci_type = _ci_type(idx.kind)
    return f(ci_type(idx.value))
end

"""
    CouplingEntry

Single (constraint_id, coefficient) pair stored in a flat array.
isbits — enables contiguous storage in `Vector{CouplingEntry}`.
"""
struct CouplingEntry
    constraint_id::TaggedCI
    coefficient::Float64
end
@assert isbitstype(CouplingEntry)

"""
    CouplingCoefficients

CSR-style storage for coupling coefficients of SP variables.
Iterating over a sp_var's coefficients is a @view into a flat array — allocation-free.
"""
struct CouplingCoefficients
    entries::Vector{CouplingEntry}
    offsets::Dict{_VI,UnitRange{Int}}
end

function CouplingCoefficients()
    return CouplingCoefficients(CouplingEntry[], Dict{_VI,UnitRange{Int}}())
end

@inline function get_coefficients(cc::CouplingCoefficients, sp_var::_VI)
    range = get(cc.offsets, sp_var, 1:0)
    return @view cc.entries[range]
end
