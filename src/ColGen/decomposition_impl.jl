# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# PART 1: COUPLING COEFFICIENT STORAGE (CSR-style, allocation-free iteration)
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    CouplingEntry{C}

Single (constraint_id, coefficient) pair stored in a flat array.
isbits when C is isbits (true for MOI.ConstraintIndex).
"""
struct CouplingEntry{C}
    constraint_id::C
    coefficient::Float64
end

"""
    CouplingCoefficients{V,C}

CSR-style storage for coupling coefficients of SP variables.
Iterating over a sp_var's coefficients is a @view into a flat array — allocation-free.
"""
struct CouplingCoefficients{V,C}
    entries::Vector{CouplingEntry{C}}
    offsets::Dict{V,UnitRange{Int}}
end

function CouplingCoefficients{V,C}() where {V,C}
    return CouplingCoefficients(CouplingEntry{C}[], Dict{V,UnitRange{Int}}())
end

@inline function get_coefficients(cc::CouplingCoefficients{V}, sp_var::V) where {V}
    range = get(cc.offsets, sp_var, 1:0)
    return @view cc.entries[range]
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 2: SUBPROBLEM DATA
# ────────────────────────────────────────────────────────────────────────────────────────

struct SubproblemData{V,C}
    variables::Vector{V}
    original_costs::Dict{V,Float64}
    coupling_coeffs::CouplingCoefficients{V,C}
    fixed_cost::Float64
    convexity_lb::Float64
    convexity_ub::Float64
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 3: PURE MASTER VARIABLE DATA
# ────────────────────────────────────────────────────────────────────────────────────────

struct PureMasterVariableData{Y,C}
    id::Y
    cost::Float64
    lb::Float64
    ub::Float64
    is_integer::Bool
    coupling_coeffs::Vector{CouplingEntry{C}}
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 4: FORWARD MAPPING M (x → z)
# ────────────────────────────────────────────────────────────────────────────────────────

struct ForwardMapping{X,S,V}
    forward::Dict{X,Vector{Tuple{S,V}}}
    inverse_set::Dict{Tuple{S,V},Vector{X}}
    all_orig_vars::Vector{X}
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 5: CONSTRAINT SENSE ENUM
# ────────────────────────────────────────────────────────────────────────────────────────

@enum ConstraintSense begin
    GREATER_THAN  # ax ≥ b
    LESS_THAN     # ax ≤ b
    EQUAL_TO      # ax = b
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 6: THE DECOMPOSITION STRUCT
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    Decomposition{S,V,X,C,Y}

Immutable concrete implementation of AbstractDecomposition.

Type parameters:
  - S: subproblem identifier type (e.g., Int)
  - V: SP variable identifier type (e.g., MOI.VariableIndex)
  - X: original/linking variable identifier type
  - C: constraint identifier type (e.g., MOI.ConstraintIndex)
  - Y: pure master variable identifier type
"""
struct Decomposition{S,V,X,C,Y} <: AbstractDecomposition
    subproblems::Dict{S,SubproblemData{V,C}}
    pure_master_vars::Vector{PureMasterVariableData{Y,C}}
    mapping::ForwardMapping{X,S,V}
    coupling_cstrs::Vector{Tuple{C,ConstraintSense,Float64}}
    minimize::Bool
end

# M⁻¹ hot path
@inline function original_cost(d::Decomposition, sp_id, sp_var)
    return d.subproblems[sp_id].original_costs[sp_var]
end

@inline function coupling_coefficients(d::Decomposition, sp_id, sp_var)
    return get_coefficients(d.subproblems[sp_id].coupling_coeffs, sp_var)
end

# M forward (cold path)
function mapped_subproblem_variables(d::Decomposition, orig_var)
    return get(d.mapping.forward, orig_var, eltype(values(d.mapping.forward))[])
end

function mapping_to_original(d::Decomposition, sp_id, sp_var)
    return get(d.mapping.inverse_set, (sp_id, sp_var), eltype(d.mapping.all_orig_vars)[])
end

original_variables(d::Decomposition) = d.mapping.all_orig_vars

# Subproblem queries
subproblem_ids(d::Decomposition) = keys(d.subproblems)

function subproblem_variables(d::Decomposition, sp_id)
    return d.subproblems[sp_id].variables
end

function subproblem_fixed_cost(d::Decomposition, sp_id)
    return d.subproblems[sp_id].fixed_cost
end

function convexity_bounds(d::Decomposition, sp_id)
    sp = d.subproblems[sp_id]
    return (sp.convexity_lb, sp.convexity_ub)
end

nb_subproblem_multiplicity(d::Decomposition, sp_id) = d.subproblems[sp_id].convexity_ub

# Pure master variable queries
pure_master_variables(d::Decomposition) = d.pure_master_vars
pure_master_cost(::Decomposition, pmv::PureMasterVariableData) = pmv.cost
pure_master_bounds(::Decomposition, pmv::PureMasterVariableData) = (pmv.lb, pmv.ub)
pure_master_is_integer(::Decomposition, pmv::PureMasterVariableData) = pmv.is_integer
function pure_master_coupling_coefficients(::Decomposition, pmv::PureMasterVariableData)
    return pmv.coupling_coeffs
end

# Master
coupling_constraints(d::Decomposition) = d.coupling_cstrs
is_minimization(d::Decomposition) = d.minimize


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 7: SUBPROBLEM SOLUTION
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    SpSolution{S,V}

Concrete subproblem solution with sorted entries for deterministic order and
cheap deduplication via fingerprint hash.

`obj_value` is the pricing subproblem objective (reduced-cost objective).
"""
struct SpSolution{S,V} <: AbstractSubproblemSolution
    sp_id::S
    obj_value::Float64
    entries::Vector{Tuple{V,Float64}}  # sorted by sp_var
    fingerprint::UInt64
end

function SpSolution(sp_id::S, obj_value::Float64, entries::Vector{Tuple{MOI.VariableIndex,Float64}}) where {S}
    sorted = sort(entries; by = e -> e[1].value)
    filter!(e -> !iszero(e[2]), sorted)
    fp = hash(map(e -> (e[1].value, round(e[2]; digits=10)), sorted))
    return SpSolution{S,MOI.VariableIndex}(sp_id, obj_value, sorted, fp)
end

subproblem_id(sol::SpSolution) = sol.sp_id
objective_value(sol::SpSolution) = sol.obj_value
@inline nonzero_entries(sol::SpSolution) = sol.entries

function solution_value(sol::SpSolution{S,V}, sp_var::V) where {S,V}
    idx = searchsortedfirst(sol.entries, (sp_var, -Inf); by=first)
    if idx <= length(sol.entries) && first(sol.entries[idx]) == sp_var
        return sol.entries[idx][2]
    end
    return 0.0
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 8: COLUMN POOL
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    ColumnRecord{S,V}

A column stored in the pool. `original_cost` is the column's cost in the
master's original objective.
"""
struct ColumnRecord{S,V}
    sp_id::S
    solution::SpSolution{S,V}
    original_cost::Float64
end

"""
    ColumnPool{C,S,V}

Triple-indexed column pool: by column variable, by subproblem, and by
fingerprint for O(1) dedup.

- `C`: column variable index type (e.g. `MOI.VariableIndex`)
- `S`: subproblem identifier type
- `V`: subproblem variable index type (e.g. `MOI.VariableIndex`)
"""
mutable struct ColumnPool{C,S,V} <: AbstractColumnPool
    by_column_var::Dict{C,ColumnRecord{S,V}}
    by_subproblem::Dict{S,Vector{C}}
    fingerprints::Dict{S,Set{UInt64}}
end

function ColumnPool{C,S,V}() where {C,S,V}
    return ColumnPool(
        Dict{C,ColumnRecord{S,V}}(),
        Dict{S,Vector{C}}(),
        Dict{S,Set{UInt64}}()
    )
end

function record_column!(
    pool::ColumnPool{C,S,V}, col_var::C, sp_id::S, sol::SpSolution{S,V}, cost::Float64
) where {C,S,V}
    pool.by_column_var[col_var] = ColumnRecord(sp_id, sol, cost)
    sp_cols = get!(Vector{C}, pool.by_subproblem, sp_id)
    push!(sp_cols, col_var)
    fp_set = get!(Set{UInt64}, pool.fingerprints, sp_id)
    push!(fp_set, sol.fingerprint)
    return nothing
end

function get_column_solution(pool::ColumnPool, col_var)
    record = get(pool.by_column_var, col_var, nothing)
    return isnothing(record) ? nothing : record.solution
end

get_column_sp_id(pool::ColumnPool, col_var) = pool.by_column_var[col_var].sp_id
get_column_cost(pool::ColumnPool, col_var) = pool.by_column_var[col_var].original_cost

function columns(pool::ColumnPool)
    return (
        (cv, rec.sp_id, rec.solution, rec.original_cost)
        for (cv, rec) in pool.by_column_var
    )
end

function columns_for_subproblem(pool::ColumnPool{C,S,V}, sp_id::S) where {C,S,V}
    col_vars = get(pool.by_subproblem, sp_id, C[])
    return (
        (cv, pool.by_column_var[cv].solution, pool.by_column_var[cv].original_cost)
        for cv in col_vars
    )
end

function has_column(pool::ColumnPool, sp_id, sol::SpSolution)
    fp_set = get(pool.fingerprints, sp_id, nothing)
    isnothing(fp_set) && return false
    return sol.fingerprint in fp_set
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 9: BUILDER
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    DecompositionBuilder{S,V,X,C,Y}

Incrementally builds an immutable Decomposition from problem data.
"""
mutable struct DecompositionBuilder{S,V,X,C,Y}
    minimize::Bool
    sp_variables::Dict{S,Vector{V}}
    sp_original_costs::Dict{S,Dict{V,Float64}}
    sp_coupling_entries::Dict{S,Vector{Tuple{V,C,Float64}}}
    sp_fixed_costs::Dict{S,Float64}
    sp_conv_bounds::Dict{S,Tuple{Float64,Float64}}
    pm_vars::Vector{PureMasterVariableData{Y,C}}
    pm_coupling_accum::Dict{Y,Vector{CouplingEntry{C}}}
    pm_data_accum::Dict{Y,Tuple{Float64,Float64,Float64,Bool}}
    forward_map::Dict{X,Vector{Tuple{S,V}}}
    inverse_map::Dict{Tuple{S,V},Vector{X}}
    all_orig_vars_set::Set{X}
    coupling_cstrs::Vector{Tuple{C,ConstraintSense,Float64}}
end

function DecompositionBuilder{S,V,X,C,Y}(; minimize::Bool=true) where {S,V,X,C,Y}
    return DecompositionBuilder{S,V,X,C,Y}(
        minimize,
        Dict{S,Vector{V}}(),
        Dict{S,Dict{V,Float64}}(),
        Dict{S,Vector{Tuple{V,C,Float64}}}(),
        Dict{S,Float64}(),
        Dict{S,Tuple{Float64,Float64}}(),
        PureMasterVariableData{Y,C}[],
        Dict{Y,Vector{CouplingEntry{C}}}(),
        Dict{Y,Tuple{Float64,Float64,Float64,Bool}}(),
        Dict{X,Vector{Tuple{S,V}}}(),
        Dict{Tuple{S,V},Vector{X}}(),
        Set{X}(),
        Tuple{C,ConstraintSense,Float64}[]
    )
end

function add_subproblem!(
    b::DecompositionBuilder{S,V,X,C,Y},
    sp_id::S,
    fixed_cost::Float64,
    conv_lb::Float64,
    conv_ub::Float64
) where {S,V,X,C,Y}
    b.sp_variables[sp_id] = V[]
    b.sp_original_costs[sp_id] = Dict{V,Float64}()
    b.sp_coupling_entries[sp_id] = Tuple{V,C,Float64}[]
    b.sp_fixed_costs[sp_id] = fixed_cost
    b.sp_conv_bounds[sp_id] = (conv_lb, conv_ub)
    return nothing
end

function add_sp_variable!(
    b::DecompositionBuilder{S,V}, sp_id::S, sp_var::V, orig_cost::Float64
) where {S,V}
    push!(b.sp_variables[sp_id], sp_var)
    b.sp_original_costs[sp_id][sp_var] = orig_cost
    return nothing
end

function add_coupling_coefficient!(
    b::DecompositionBuilder{S,V,X,C}, sp_id::S, sp_var::V, cstr_id::C, coeff::Float64
) where {S,V,X,C}
    push!(get!(Vector{Tuple{V,C,Float64}}, b.sp_coupling_entries, sp_id), (sp_var, cstr_id, coeff))
    return nothing
end

function add_mapping!(
    b::DecompositionBuilder{S,V,X}, orig_var::X, sp_id::S, sp_var::V
) where {S,V,X}
    push!(get!(Vector{Tuple{S,V}}, b.forward_map, orig_var), (sp_id, sp_var))
    push!(get!(Vector{X}, b.inverse_map, (sp_id, sp_var)), orig_var)
    push!(b.all_orig_vars_set, orig_var)
    return nothing
end

function add_pure_master_variable!(
    b::DecompositionBuilder{S,V,X,C,Y},
    y_id::Y,
    cost::Float64,
    lb::Float64,
    ub::Float64,
    is_integer::Bool
) where {S,V,X,C,Y}
    b.pm_data_accum[y_id] = (cost, lb, ub, is_integer)
    b.pm_coupling_accum[y_id] = CouplingEntry{C}[]
    return nothing
end

function add_pure_master_coupling!(
    b::DecompositionBuilder{S,V,X,C,Y}, y_id::Y, cstr_id::C, coeff::Float64
) where {S,V,X,C,Y}
    push!(b.pm_coupling_accum[y_id], CouplingEntry(cstr_id, coeff))
    return nothing
end

function add_coupling_constraint!(
    b::DecompositionBuilder{S,V,X,C}, cstr_id::C, sense::ConstraintSense, rhs::Float64
) where {S,V,X,C}
    push!(b.coupling_cstrs, (cstr_id, sense, rhs))
    return nothing
end

"""
    build(builder) -> Decomposition

Compile the accumulated data into an immutable Decomposition.
Builds the CSR coupling coefficient structure in O(total entries).
"""
function build(b::DecompositionBuilder{S,V,X,C,Y}) where {S,V,X,C,Y}
    subproblems = Dict{S,SubproblemData{V,C}}()

    for sp_id in keys(b.sp_variables)
        variables = b.sp_variables[sp_id]
        original_costs = b.sp_original_costs[sp_id]

        entries = CouplingEntry{C}[]
        offsets = Dict{V,UnitRange{Int}}()

        # Group raw entries by sp_var using a dict (avoids requiring isless on V)
        raw_entries = get(b.sp_coupling_entries, sp_id, Tuple{V,C,Float64}[])
        grouped = Dict{V,Vector{Tuple{C,Float64}}}()
        for (sp_var, cstr_id, coeff) in raw_entries
            grp = get!(Vector{Tuple{C,Float64}}, grouped, sp_var)
            push!(grp, (cstr_id, coeff))
        end

        # Build CSR in the same order as the variables list (deterministic)
        for sp_var in variables
            haskey(grouped, sp_var) || continue
            range_start = length(entries) + 1
            for (cstr_id, coeff) in grouped[sp_var]
                push!(entries, CouplingEntry(cstr_id, coeff))
            end
            offsets[sp_var] = range_start:length(entries)
        end

        coupling_coeffs = CouplingCoefficients(entries, offsets)
        conv_lb, conv_ub = b.sp_conv_bounds[sp_id]

        subproblems[sp_id] = SubproblemData(
            variables, original_costs, coupling_coeffs,
            b.sp_fixed_costs[sp_id], conv_lb, conv_ub
        )
    end

    pm_vars = PureMasterVariableData{Y,C}[]
    for (y_id, (cost, lb, ub, is_int)) in b.pm_data_accum
        coeffs = get(b.pm_coupling_accum, y_id, CouplingEntry{C}[])
        push!(pm_vars, PureMasterVariableData(y_id, cost, lb, ub, is_int, coeffs))
    end

    mapping = ForwardMapping(
        b.forward_map,
        b.inverse_map,
        collect(b.all_orig_vars_set)
    )

    return Decomposition{S,V,X,C,Y}(
        subproblems, pm_vars, mapping, b.coupling_cstrs, b.minimize
    )
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 10: NON-ROBUST CUT MANAGER (stub — returns 0 for all cuts)
# ────────────────────────────────────────────────────────────────────────────────────────

struct ActiveNonRobustCut{C}
    master_constraint::C
    family::AbstractNonRobustCutFamily
    data::Any
end

mutable struct NonRobustCutManager{C}
    cuts::Vector{ActiveNonRobustCut{C}}
    duals::Dict{C,Float64}
end

NonRobustCutManager{C}() where {C} = NonRobustCutManager(
    ActiveNonRobustCut{C}[], Dict{C,Float64}()
)

function add_cut!(
    mgr::NonRobustCutManager{C},
    master_cstr::C,
    family::AbstractNonRobustCutFamily,
    data
) where {C}
    push!(mgr.cuts, ActiveNonRobustCut(master_cstr, family, data))
    mgr.duals[master_cstr] = 0.0
    return nothing
end

function update_duals!(mgr::NonRobustCutManager, dual_solution)
    for cut in mgr.cuts
        mgr.duals[cut.master_constraint] = dual_solution(cut.master_constraint)
    end
    return nothing
end

function total_cut_dual_contribution(mgr::NonRobustCutManager, sp_id, sp_var)
    total = 0.0
    for cut in mgr.cuts
        σ = mgr.duals[cut.master_constraint]
        iszero(σ) && continue
        total += compute_cut_dual_contribution(cut.family, cut.data, σ, sp_id, sp_var)
    end
    return total
end

function compute_column_cut_coefficients(
    mgr::NonRobustCutManager{C}, sol::AbstractSubproblemSolution
) where {C}
    coeffs = Dict{C,Float64}()
    for cut in mgr.cuts
        coeff = compute_cut_coefficient(cut.family, cut.data, sol)
        if !iszero(coeff)
            coeffs[cut.master_constraint] = coeff
        end
    end
    return coeffs
end
