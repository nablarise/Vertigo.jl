# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct SubproblemData
    variables::Vector{_VI}
    original_costs::Dict{_VI,Float64}
    coupling_coeffs::CouplingCoefficients
    fixed_cost::Float64
    convexity_lb::Float64
    convexity_ub::Float64
end

struct PureMasterVariableData
    id::_VI
    cost::Float64
    lb::Float64
    ub::Float64
    is_integer::Bool
    coupling_coeffs::Vector{CouplingEntry}
end

"""
    AbstractVariableMapping{X}

Abstract supertype for variable mappings between original and subproblem spaces.
Allows dispatch-based specialization for different mapping cardinalities.
"""
abstract type AbstractVariableMapping{X} end

"""
    OneToOneMapping{X} <: AbstractVariableMapping{X}

Bidirectional 1:1 mapping between original variables and subproblem variables.

`X` is the original variable identifier type (e.g. `Tuple{Int,Int}`
for a (machine, task) pair in GAP).

# Fields
- `forward`: original var → (subproblem_id, sp_var).
- `inverse`: (subproblem_id, sp_var) → original var.
- `all_orig_vars`: every original variable registered in the mapping.
"""
struct OneToOneMapping{X} <: AbstractVariableMapping{X}
    forward::Dict{X,Tuple{PricingSubproblemId,_VI}}
    inverse::Dict{Tuple{PricingSubproblemId,_VI},X}
    all_orig_vars::Vector{X}
end
