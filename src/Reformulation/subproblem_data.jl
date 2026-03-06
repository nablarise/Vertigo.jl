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
    ForwardMapping{X}

Bidirectional mapping between original variables and subproblem
variables.

`X` is the original variable identifier type (e.g. `Tuple{Int,Int}`
for a (machine, task) pair in GAP).

# Fields
- `forward`: original var → subproblem copies it maps to.
- `inverse_set`: (subproblem, sp_var) → original vars it represents.
- `all_orig_vars`: every original variable registered in the mapping.
"""
struct ForwardMapping{X}
    forward::Dict{X,Vector{Tuple{PricingSubproblemId,_VI}}}
    inverse_set::Dict{Tuple{PricingSubproblemId,_VI},Vector{X}}
    all_orig_vars::Vector{X}
end
