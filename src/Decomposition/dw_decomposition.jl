# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    DWDecomposition{X,M}

Mutable concrete implementation of AbstractDecomposition.

Type parameters:
  - X: original/linking variable identifier type
  - M: variable mapping type (<: AbstractVariableMapping{X})
"""
mutable struct DWDecomposition{X,M<:AbstractVariableMapping{X}} <: AbstractDecomposition
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
@inline function original_cost(d::DWDecomposition, sp_id, sp_var)
    return d.subproblems[sp_id].original_costs[sp_var]
end

@inline function coupling_coefficients(d::DWDecomposition, sp_id, sp_var)
    return get_coefficients(d.subproblems[sp_id].coupling_coeffs, sp_var)
end

# M⁻¹ : (sp_id, sp_var) → original var or nothing
function mapped_original_var(d::DWDecomposition, sp_id, sp_var)
    return _mapped_original_var(d.mapping, sp_id, sp_var)
end
function _mapped_original_var(m::OneToOneMapping, sp_id, sp_var)
    return get(m.inverse, (sp_id, sp_var), nothing)
end

# M : orig_var → (sp_id, sp_var) or nothing
function mapped_subproblem_var(d::DWDecomposition, orig_var)
    return _mapped_subproblem_var(d.mapping, orig_var)
end
_mapped_subproblem_var(m::OneToOneMapping, orig_var) = get(m.forward, orig_var, nothing)

original_variables(d::DWDecomposition) = d.mapping.all_orig_vars

orig_var_type(::DWDecomposition{X}) where {X} = X

# Subproblem queries
subproblem_ids(d::DWDecomposition) = keys(d.subproblems)

function subproblem_variables(d::DWDecomposition, sp_id)
    return d.subproblems[sp_id].variables
end

function subproblem_fixed_cost(d::DWDecomposition, sp_id)
    return d.subproblems[sp_id].fixed_cost
end

function convexity_bounds(d::DWDecomposition, sp_id)
    sp = d.subproblems[sp_id]
    return (sp.convexity_lb, sp.convexity_ub)
end

nb_subproblem_multiplicity(d::DWDecomposition, sp_id) = d.subproblems[sp_id].convexity_ub

# Pure master variable queries
pure_master_variables(d::DWDecomposition) = d.pure_master_vars
pure_master_cost(::DWDecomposition, pmv::PureMasterVariableData) = pmv.cost
pure_master_bounds(::DWDecomposition, pmv::PureMasterVariableData) = (pmv.lb, pmv.ub)
pure_master_is_integer(::DWDecomposition, pmv::PureMasterVariableData) = pmv.is_integer
function pure_master_coupling_coefficients(::DWDecomposition, pmv::PureMasterVariableData)
    return pmv.coupling_coeffs
end

# Master
coupling_constraints(d::DWDecomposition) = d.coupling_cstrs
is_minimization(d::DWDecomposition) = d.minimize

# MOI model accessors
master_model(d::DWDecomposition) = d.master_model
sp_model(d::DWDecomposition, sp_id) = d.sp_models[sp_id]
sp_models(d::DWDecomposition) = d.sp_models
convexity_ub_ci(d::DWDecomposition, sp_id) = d.convexity_ub[sp_id]
convexity_lb_ci(d::DWDecomposition, sp_id) = d.convexity_lb[sp_id]
has_convexity_ub(d::DWDecomposition, sp_id) = haskey(d.convexity_ub, sp_id)
has_convexity_lb(d::DWDecomposition, sp_id) = haskey(d.convexity_lb, sp_id)
convexity_ub_pairs(d::DWDecomposition) = d.convexity_ub
convexity_lb_pairs(d::DWDecomposition) = d.convexity_lb

function set_models!(d::DWDecomposition, master, sps, conv_ub, conv_lb)
    d.master_model = master
    d.sp_models = sps
    d.convexity_ub = conv_ub
    d.convexity_lb = conv_lb
    return nothing
end

"""
    add_convexity_constraints!(d::DWDecomposition, master) -> (conv_ub, conv_lb)

Add to the MOI model `master` the two convexity constraints of every subproblem of `d`: the
sum of the weights of its columns is at least the lower convexity bound and at most the upper
one, as given by `convexity_bounds`. The constraints have no column yet. Returns the
dictionaries of the upper and lower constraints, in the order expected by `set_models!`.
"""
function add_convexity_constraints!(d::DWDecomposition, master)
    conv_ub = Dict{PricingSubproblemId,TaggedCI}()
    conv_lb = Dict{PricingSubproblemId,TaggedCI}()
    for sp_id in sort(collect(subproblem_ids(d)); by=id -> id.value)
        lb, ub = convexity_bounds(d, sp_id)
        conv_lb[sp_id] = TaggedCI(
            MOI.add_constraint(
                master, _SAF(MOI.ScalarAffineTerm{Float64}[], 0.0), MOI.GreaterThan(lb)
            ),
        )
        conv_ub[sp_id] = TaggedCI(
            MOI.add_constraint(
                master, _SAF(MOI.ScalarAffineTerm{Float64}[], 0.0), MOI.LessThan(ub)
            ),
        )
    end
    return conv_ub, conv_lb
end

# Typed specialization: returns Dict{X,Float64} instead of Dict{Any,Float64}
function project_to_original(
    decomp::DWDecomposition{X}, pool::AbstractColumnPool, master_primal_values
) where {X}
    x_values = Dict{X,Float64}()
    for (col_var, rec) in columns(pool)
        λ_val = master_primal_values(col_var)
        if iszero(λ_val)
            continue
        end
        sp_id = column_sp_id(rec)
        for (sp_var, z_val) in column_nonzero_entries(rec)
            orig_var = mapped_original_var(decomp, sp_id, sp_var)
            if orig_var !== nothing
                x_values[orig_var] = get(x_values, orig_var, 0.0) + z_val * λ_val
            end
        end
    end
    return x_values
end
