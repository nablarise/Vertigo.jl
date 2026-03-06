# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    DWReformulation{X}

Immutable concrete implementation of AbstractDecomposition.

Type parameters:
  - X: original/linking variable identifier type
"""
struct DWReformulation{X} <: AbstractDecomposition
    subproblems::Dict{PricingSubproblemId,SubproblemData}
    pure_master_vars::Vector{PureMasterVariableData}
    mapping::ForwardMapping{X}
    coupling_cstrs::Vector{Tuple{TaggedCI,Float64}}
    minimize::Bool
end

# M⁻¹ hot path
@inline function original_cost(d::DWReformulation, sp_id, sp_var)
    return d.subproblems[sp_id].original_costs[sp_var]
end

@inline function coupling_coefficients(d::DWReformulation, sp_id, sp_var)
    return get_coefficients(d.subproblems[sp_id].coupling_coeffs, sp_var)
end

# M forward (cold path)
function mapped_subproblem_variables(d::DWReformulation, orig_var)
    return get(d.mapping.forward, orig_var, eltype(values(d.mapping.forward))[])
end

function mapping_to_original(d::DWReformulation, sp_id, sp_var)
    return get(d.mapping.inverse_set, (sp_id, sp_var), eltype(d.mapping.all_orig_vars)[])
end

original_variables(d::DWReformulation) = d.mapping.all_orig_vars

# Subproblem queries
subproblem_ids(d::DWReformulation) = keys(d.subproblems)

function subproblem_variables(d::DWReformulation, sp_id)
    return d.subproblems[sp_id].variables
end

function subproblem_fixed_cost(d::DWReformulation, sp_id)
    return d.subproblems[sp_id].fixed_cost
end

function convexity_bounds(d::DWReformulation, sp_id)
    sp = d.subproblems[sp_id]
    return (sp.convexity_lb, sp.convexity_ub)
end

nb_subproblem_multiplicity(d::DWReformulation, sp_id) = d.subproblems[sp_id].convexity_ub

# Pure master variable queries
pure_master_variables(d::DWReformulation) = d.pure_master_vars
pure_master_cost(::DWReformulation, pmv::PureMasterVariableData) = pmv.cost
pure_master_bounds(::DWReformulation, pmv::PureMasterVariableData) = (pmv.lb, pmv.ub)
pure_master_is_integer(::DWReformulation, pmv::PureMasterVariableData) = pmv.is_integer
function pure_master_coupling_coefficients(::DWReformulation, pmv::PureMasterVariableData)
    return pmv.coupling_coeffs
end

# Master
coupling_constraints(d::DWReformulation) = d.coupling_cstrs
is_minimization(d::DWReformulation) = d.minimize
