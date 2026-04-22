# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    DWReformulation{X,M}

Mutable concrete implementation of AbstractDecomposition.

Type parameters:
  - X: original/linking variable identifier type
  - M: variable mapping type (<: AbstractVariableMapping{X})
"""
mutable struct DWReformulation{X,M<:AbstractVariableMapping{X}} <: AbstractDecomposition
    subproblems::Dict{PricingSubproblemId,SubproblemData}
    pure_master_vars::Vector{PureMasterVariableData}
    mapping::M
    coupling_cstrs::Vector{Tuple{TaggedCI,Float64}}
    minimize::Bool
    master_model::Any
    sp_models::Dict{PricingSubproblemId,Any}
    convexity_ub::Dict{PricingSubproblemId,TaggedCI}
    convexity_lb::Dict{PricingSubproblemId,TaggedCI}
end

# M⁻¹ hot path
@inline function original_cost(d::DWReformulation, sp_id, sp_var)
    return d.subproblems[sp_id].original_costs[sp_var]
end

@inline function coupling_coefficients(d::DWReformulation, sp_id, sp_var)
    return get_coefficients(d.subproblems[sp_id].coupling_coeffs, sp_var)
end

# M⁻¹ : (sp_id, sp_var) → original var or nothing
mapped_original_var(d::DWReformulation, sp_id, sp_var) =
    _mapped_original_var(d.mapping, sp_id, sp_var)
_mapped_original_var(m::OneToOneMapping, sp_id, sp_var) =
    get(m.inverse, (sp_id, sp_var), nothing)

# M : orig_var → (sp_id, sp_var) or nothing
mapped_subproblem_var(d::DWReformulation, orig_var) =
    _mapped_subproblem_var(d.mapping, orig_var)
_mapped_subproblem_var(m::OneToOneMapping, orig_var) =
    get(m.forward, orig_var, nothing)

original_variables(d::DWReformulation) = d.mapping.all_orig_vars

orig_var_type(::DWReformulation{X}) where {X} = X

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

# MOI model accessors
master_model(d::DWReformulation) = d.master_model
sp_model(d::DWReformulation, sp_id) = d.sp_models[sp_id]
sp_models(d::DWReformulation) = d.sp_models
convexity_ub_ci(d::DWReformulation, sp_id) = d.convexity_ub[sp_id]
convexity_lb_ci(d::DWReformulation, sp_id) = d.convexity_lb[sp_id]
has_convexity_ub(d::DWReformulation, sp_id) = haskey(d.convexity_ub, sp_id)
has_convexity_lb(d::DWReformulation, sp_id) = haskey(d.convexity_lb, sp_id)
convexity_ub_pairs(d::DWReformulation) = d.convexity_ub
convexity_lb_pairs(d::DWReformulation) = d.convexity_lb

function set_models!(
    d::DWReformulation, master, sps, conv_ub, conv_lb
)
    d.master_model = master
    d.sp_models = sps
    d.convexity_ub = conv_ub
    d.convexity_lb = conv_lb
    return nothing
end

# Typed specialization: returns Dict{X,Float64} instead of Dict{Any,Float64}
function project_to_original(
    decomp::DWReformulation{X}, pool::AbstractColumnPool,
    master_primal_values
) where {X}
    x_values = Dict{X,Float64}()
    for (col_var, rec) in columns(pool)
        λ_val = master_primal_values(col_var)
        iszero(λ_val) && continue
        sp_id = column_sp_id(rec)
        for (sp_var, z_val) in column_nonzero_entries(rec)
            orig_var = mapped_original_var(decomp, sp_id, sp_var)
            if orig_var !== nothing
                x_values[orig_var] = get(
                    x_values, orig_var, 0.0
                ) + z_val * λ_val
            end
        end
    end
    return x_values
end
