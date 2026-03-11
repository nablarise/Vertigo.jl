# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    AbstractCutSeparator

Interface for robust cut separators. Implement `separate(s, x)` to
return a vector of `SeparatedCut` from a fractional solution `x`.
"""
abstract type AbstractCutSeparator end

"""
    SeparatedCut

A robust cut expressed as a linear function of original variables.

# Fields
- `coefficients`: mapping from original variable to coefficient.
- `set`: constraint set (`LessThan`, `GreaterThan`, or `EqualTo`).
"""
struct SeparatedCut
    coefficients::Dict{Any,Float64}
    set::Union{
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.EqualTo{Float64}
    }
end

"""
    separate(separator, x) -> Vector{SeparatedCut}

Separate robust cuts violated by the fractional solution `x`.
`x` is a `Dict{Any,Float64}` mapping original variables to values.
"""
function separate end

"""
    CustomCutSeparator <: AbstractCutSeparator

Wraps a user callback for cut separation.

# Fields
- `callback`: function `(x::Dict{Any,Float64}) -> Vector{SeparatedCut}`.
"""
struct CustomCutSeparator <: AbstractCutSeparator
    callback::Function
end

separate(s::CustomCutSeparator, x) = s.callback(x)
