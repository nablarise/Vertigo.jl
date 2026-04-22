# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    DWReformulationBuilder{X}

Incrementally builds an immutable DWReformulation from problem data.
"""
mutable struct DWReformulationBuilder{X}
    minimize::Bool
    sp_variables::Dict{PricingSubproblemId,Vector{_VI}}
    sp_original_costs::Dict{PricingSubproblemId,Dict{_VI,Float64}}
    sp_coupling_entries::Dict{PricingSubproblemId,Vector{Tuple{_VI,TaggedCI,Float64}}}
    sp_fixed_costs::Dict{PricingSubproblemId,Float64}
    sp_conv_bounds::Dict{PricingSubproblemId,Tuple{Float64,Float64}}
    pm_vars::Vector{PureMasterVariableData}
    pm_coupling_accum::Dict{_VI,Vector{CouplingEntry}}
    pm_data_accum::Dict{_VI,Tuple{Float64,Float64,Float64,Bool}}
    forward_map::Dict{X,Tuple{PricingSubproblemId,_VI}}
    inverse_map::Dict{Tuple{PricingSubproblemId,_VI},X}
    all_orig_vars_set::Set{X}
    coupling_cstrs::Vector{Tuple{TaggedCI,Float64}}
end

function DWReformulationBuilder{X}(; minimize::Bool=true) where {X}
    return DWReformulationBuilder{X}(
        minimize,
        Dict{PricingSubproblemId,Vector{_VI}}(),
        Dict{PricingSubproblemId,Dict{_VI,Float64}}(),
        Dict{PricingSubproblemId,Vector{Tuple{_VI,TaggedCI,Float64}}}(),
        Dict{PricingSubproblemId,Float64}(),
        Dict{PricingSubproblemId,Tuple{Float64,Float64}}(),
        PureMasterVariableData[],
        Dict{_VI,Vector{CouplingEntry}}(),
        Dict{_VI,Tuple{Float64,Float64,Float64,Bool}}(),
        Dict{X,Tuple{PricingSubproblemId,_VI}}(),
        Dict{Tuple{PricingSubproblemId,_VI},X}(),
        Set{X}(),
        Tuple{TaggedCI,Float64}[]
    )
end

function add_subproblem!(
    b::DWReformulationBuilder{X},
    sp_id::PricingSubproblemId,
    fixed_cost::Float64,
    conv_lb::Float64,
    conv_ub::Float64
) where {X}
    b.sp_variables[sp_id] = _VI[]
    b.sp_original_costs[sp_id] = Dict{_VI,Float64}()
    b.sp_coupling_entries[sp_id] = Tuple{_VI,TaggedCI,Float64}[]
    b.sp_fixed_costs[sp_id] = fixed_cost
    b.sp_conv_bounds[sp_id] = (conv_lb, conv_ub)
    return nothing
end

function add_sp_variable!(
    b::DWReformulationBuilder, sp_id::PricingSubproblemId,
    sp_var::_VI, orig_cost::Float64
)
    push!(b.sp_variables[sp_id], sp_var)
    b.sp_original_costs[sp_id][sp_var] = orig_cost
    return nothing
end

function add_coupling_coefficient!(
    b::DWReformulationBuilder, sp_id::PricingSubproblemId,
    sp_var::_VI, cstr_id, coeff::Float64
)
    tagged = TaggedCI(cstr_id)
    push!(get!(Vector{Tuple{_VI,TaggedCI,Float64}}, b.sp_coupling_entries, sp_id), (sp_var, tagged, coeff))
    return nothing
end

function add_mapping!(
    b::DWReformulationBuilder{X}, orig_var::X,
    sp_id::PricingSubproblemId, sp_var::_VI
) where {X}
    @assert !haskey(b.forward_map, orig_var) "Duplicate forward mapping for $orig_var"
    @assert !haskey(b.inverse_map, (sp_id, sp_var)) "Duplicate inverse mapping for ($sp_id, $sp_var)"
    b.forward_map[orig_var] = (sp_id, sp_var)
    b.inverse_map[(sp_id, sp_var)] = orig_var
    push!(b.all_orig_vars_set, orig_var)
    return nothing
end

function add_pure_master_variable!(
    b::DWReformulationBuilder,
    y_id::_VI,
    cost::Float64,
    lb::Float64,
    ub::Float64,
    is_integer::Bool
)
    b.pm_data_accum[y_id] = (cost, lb, ub, is_integer)
    b.pm_coupling_accum[y_id] = CouplingEntry[]
    return nothing
end

function add_pure_master_coupling!(
    b::DWReformulationBuilder, y_id::_VI, cstr_id,
    coeff::Float64
)
    push!(b.pm_coupling_accum[y_id], CouplingEntry(TaggedCI(cstr_id), coeff))
    return nothing
end

function add_coupling_constraint!(
    b::DWReformulationBuilder, cstr_id, rhs::Float64
)
    push!(b.coupling_cstrs, (TaggedCI(cstr_id), rhs))
    return nothing
end

"""
    build(builder) -> DWReformulation

Compile the accumulated data into an immutable DWReformulation.
Builds the CSR coupling coefficient structure in O(total entries).
"""
function _build_coupling_coefficients(
    variables::Vector{_VI},
    raw_entries::Vector{Tuple{_VI,TaggedCI,Float64}}
)
    entries = CouplingEntry[]
    offsets = Dict{_VI,UnitRange{Int}}()

    grouped = Dict{_VI,Vector{Tuple{TaggedCI,Float64}}}()
    for (sp_var, cstr_id, coeff) in raw_entries
        grp = get!(Vector{Tuple{TaggedCI,Float64}}, grouped, sp_var)
        push!(grp, (cstr_id, coeff))
    end

    # Build CSR in variable-list order (deterministic).
    # Each slice sorted by TaggedCI for merge-based RC computation.
    for sp_var in variables
        haskey(grouped, sp_var) || continue
        range_start = length(entries) + 1
        for (cstr_id, coeff) in grouped[sp_var]
            push!(entries, CouplingEntry(cstr_id, coeff))
        end
        sort!(@view(entries[range_start:end]); by = e -> e.constraint_id)
        offsets[sp_var] = range_start:length(entries)
    end

    return CouplingCoefficients(entries, offsets)
end

function build(b::DWReformulationBuilder{X}) where {X}
    subproblems = Dict{PricingSubproblemId,SubproblemData}()

    for sp_id in keys(b.sp_variables)
        variables = b.sp_variables[sp_id]
        original_costs = b.sp_original_costs[sp_id]
        raw_entries = get(
            b.sp_coupling_entries, sp_id,
            Tuple{_VI,TaggedCI,Float64}[]
        )
        coupling_coeffs = _build_coupling_coefficients(
            variables, raw_entries
        )
        conv_lb, conv_ub = b.sp_conv_bounds[sp_id]

        subproblems[sp_id] = SubproblemData(
            variables, original_costs, coupling_coeffs,
            b.sp_fixed_costs[sp_id], conv_lb, conv_ub
        )
    end

    pm_vars = PureMasterVariableData[]
    for (y_id, (cost, lb, ub, is_int)) in b.pm_data_accum
        coeffs = get(b.pm_coupling_accum, y_id, CouplingEntry[])
        push!(pm_vars, PureMasterVariableData(y_id, cost, lb, ub, is_int, coeffs))
    end

    mapping = OneToOneMapping(
        b.forward_map,
        b.inverse_map,
        collect(b.all_orig_vars_set)
    )

    return DWReformulation(
        subproblems, pm_vars, mapping, b.coupling_cstrs, b.minimize,
        nothing,
        Dict{PricingSubproblemId,Any}(),
        Dict{PricingSubproblemId,TaggedCI}(),
        Dict{PricingSubproblemId,TaggedCI}()
    )
end
