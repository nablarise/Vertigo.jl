# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# PART 1: COUPLING COEFFICIENT STORAGE (CSR-style, allocation-free iteration)
# ────────────────────────────────────────────────────────────────────────────────────────

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
    CouplingCoefficients{V}

CSR-style storage for coupling coefficients of SP variables.
Iterating over a sp_var's coefficients is a @view into a flat array — allocation-free.
"""
struct CouplingCoefficients{V}
    entries::Vector{CouplingEntry}
    offsets::Dict{V,UnitRange{Int}}
end

function CouplingCoefficients{V}() where {V}
    return CouplingCoefficients(CouplingEntry[], Dict{V,UnitRange{Int}}())
end

@inline function get_coefficients(cc::CouplingCoefficients{V}, sp_var::V) where {V}
    range = get(cc.offsets, sp_var, 1:0)
    return @view cc.entries[range]
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 2: SUBPROBLEM DATA
# ────────────────────────────────────────────────────────────────────────────────────────

struct SubproblemData{V}
    variables::Vector{V}
    original_costs::Dict{V,Float64}
    coupling_coeffs::CouplingCoefficients{V}
    fixed_cost::Float64
    convexity_lb::Float64
    convexity_ub::Float64
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 3: PURE MASTER VARIABLE DATA
# ────────────────────────────────────────────────────────────────────────────────────────

struct PureMasterVariableData{Y}
    id::Y
    cost::Float64
    lb::Float64
    ub::Float64
    is_integer::Bool
    coupling_coeffs::Vector{CouplingEntry}
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 4: FORWARD MAPPING M (x → z)
# ────────────────────────────────────────────────────────────────────────────────────────

struct ForwardMapping{X,V}
    forward::Dict{X,Vector{Tuple{PricingSubproblemId,V}}}
    inverse_set::Dict{Tuple{PricingSubproblemId,V},Vector{X}}
    all_orig_vars::Vector{X}
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 5: THE DECOMPOSITION STRUCT
# ────────────────────────────────────────────────────────────────────────────────────────

@enum ConstraintSense begin
    GREATER_THAN  # ax ≥ b
    LESS_THAN     # ax ≤ b
    EQUAL_TO      # ax = b
end

"""
    Decomposition{V,X,C,Y}

Immutable concrete implementation of AbstractDecomposition.

Type parameters:
  - V: SP variable identifier type (e.g., MOI.VariableIndex)
  - X: original/linking variable identifier type
  - C: constraint identifier type (e.g., MOI.ConstraintIndex)
  - Y: pure master variable identifier type
"""
struct Decomposition{V,X,C,Y} <: AbstractDecomposition
    subproblems::Dict{PricingSubproblemId,SubproblemData{V}}
    pure_master_vars::Vector{PureMasterVariableData{Y}}
    mapping::ForwardMapping{X,V}
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
    _SpSolution{V}

Concrete subproblem solution with sorted entries for deterministic order and
cheap deduplication via fingerprint hash.

`obj_value` is the pricing subproblem objective (reduced-cost objective).
"""
struct _SpSolution{V} <: AbstractSubproblemSolution
    sp_id::PricingSubproblemId
    obj_value::Float64
    entries::Vector{Tuple{V,Float64}}  # sorted by sp_var
    fingerprint::UInt64
end

function _SpSolution(sp_id::PricingSubproblemId, obj_value::Float64, entries::Vector{Tuple{MOI.VariableIndex,Float64}})
    sorted = sort(entries; by = e -> e[1].value)
    filter!(e -> !iszero(e[2]), sorted)
    fp = hash(map(e -> (e[1].value, round(e[2]; digits=10)), sorted))
    return _SpSolution{MOI.VariableIndex}(sp_id, obj_value, sorted, fp)
end

subproblem_id(sol::_SpSolution) = sol.sp_id
objective_value(sol::_SpSolution) = sol.obj_value
@inline nonzero_entries(sol::_SpSolution) = sol.entries

function solution_value(sol::_SpSolution{V}, sp_var::V) where {V}
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
    ColumnRecord{V}

A column stored in the pool. `original_cost` is the column's cost in the
master's original objective.
"""
struct ColumnRecord{V}
    sp_id::PricingSubproblemId
    solution::_SpSolution{V}
    original_cost::Float64
end

column_sp_id(rec::ColumnRecord) = rec.sp_id
column_original_cost(rec::ColumnRecord) = rec.original_cost
pricing_objective_value(rec::ColumnRecord) = rec.solution.obj_value
column_nonzero_entries(rec::ColumnRecord) = rec.solution.entries

"""
    ColumnPool{C,V}

Triple-indexed column pool: by column variable, by subproblem, and by
fingerprint for O(1) dedup.

- `C`: column variable index type (e.g. `MOI.VariableIndex`)
- `V`: subproblem variable index type (e.g. `MOI.VariableIndex`)
"""
mutable struct ColumnPool{C,V} <: AbstractColumnPool
    by_column_var::Dict{C,ColumnRecord{V}}
    by_subproblem::Dict{PricingSubproblemId,Vector{C}}
    fingerprints::Dict{PricingSubproblemId,Set{UInt64}}
end

function ColumnPool{C,V}() where {C,V}
    return ColumnPool(
        Dict{C,ColumnRecord{V}}(),
        Dict{PricingSubproblemId,Vector{C}}(),
        Dict{PricingSubproblemId,Set{UInt64}}()
    )
end

"""
    record_column!(pool, col_var, sp_id, sol, original_cost)

Register a column in the pool, indexing it by column variable, subproblem,
and fingerprint.
"""
function record_column!(
    pool::ColumnPool{C,V}, col_var::C, sp_id::PricingSubproblemId,
    sol::_SpSolution{V}, original_cost::Float64
) where {C,V}
    pool.by_column_var[col_var] = ColumnRecord(sp_id, sol, original_cost)
    sp_cols = get!(Vector{C}, pool.by_subproblem, sp_id)
    push!(sp_cols, col_var)
    fp_set = get!(Set{UInt64}, pool.fingerprints, sp_id)
    push!(fp_set, sol.fingerprint)
    return nothing
end

"""
    get_column_solution(pool, col_var) -> _SpSolution or nothing

Return the subproblem solution associated with `col_var`, or `nothing`.
"""
function get_column_solution(pool::ColumnPool, col_var)
    record = get(pool.by_column_var, col_var, nothing)
    return isnothing(record) ? nothing : record.solution
end

"Return the subproblem id of the column associated with `col_var`."
get_column_sp_id(pool::ColumnPool, col_var) = column_sp_id(pool.by_column_var[col_var])

"Return the original cost of the column associated with `col_var`."
get_column_cost(pool::ColumnPool, col_var) = column_original_cost(pool.by_column_var[col_var])

"""
    columns(pool) -> iterator of (col_var, ColumnRecord)

Iterate over all columns in the pool.
"""
function columns(pool::ColumnPool)
    return pool.by_column_var
end

"""
    columns_for_subproblem(pool, sp_id) -> iterator of (col_var, ColumnRecord)

Iterate over columns belonging to subproblem `sp_id`.
"""
function columns_for_subproblem(pool::ColumnPool{C,V}, sp_id::PricingSubproblemId) where {C,V}
    col_vars = get(pool.by_subproblem, sp_id, C[])
    return (
        (cv, pool.by_column_var[cv])
        for cv in col_vars
    )
end

"""
    has_column(pool, sp_id, sol) -> Bool

Check if a column with the same fingerprint already exists for `sp_id`.
"""
function has_column(pool::ColumnPool, sp_id, sol::_SpSolution)
    fp_set = get(pool.fingerprints, sp_id, nothing)
    isnothing(fp_set) && return false
    return sol.fingerprint in fp_set
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PART 9: BUILDER
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    DecompositionBuilder{V,X,C,Y}

Incrementally builds an immutable Decomposition from problem data.
"""
mutable struct DecompositionBuilder{V,X,C,Y}
    minimize::Bool
    sp_variables::Dict{PricingSubproblemId,Vector{V}}
    sp_original_costs::Dict{PricingSubproblemId,Dict{V,Float64}}
    sp_coupling_entries::Dict{PricingSubproblemId,Vector{Tuple{V,C,Float64}}}
    sp_fixed_costs::Dict{PricingSubproblemId,Float64}
    sp_conv_bounds::Dict{PricingSubproblemId,Tuple{Float64,Float64}}
    pm_vars::Vector{PureMasterVariableData{Y}}
    pm_coupling_accum::Dict{Y,Vector{CouplingEntry}}
    pm_data_accum::Dict{Y,Tuple{Float64,Float64,Float64,Bool}}
    forward_map::Dict{X,Vector{Tuple{PricingSubproblemId,V}}}
    inverse_map::Dict{Tuple{PricingSubproblemId,V},Vector{X}}
    all_orig_vars_set::Set{X}
    coupling_cstrs::Vector{Tuple{C,ConstraintSense,Float64}}
end

function DecompositionBuilder{V,X,C,Y}(; minimize::Bool=true) where {V,X,C,Y}
    return DecompositionBuilder{V,X,C,Y}(
        minimize,
        Dict{PricingSubproblemId,Vector{V}}(),
        Dict{PricingSubproblemId,Dict{V,Float64}}(),
        Dict{PricingSubproblemId,Vector{Tuple{V,C,Float64}}}(),
        Dict{PricingSubproblemId,Float64}(),
        Dict{PricingSubproblemId,Tuple{Float64,Float64}}(),
        PureMasterVariableData{Y}[],
        Dict{Y,Vector{CouplingEntry}}(),
        Dict{Y,Tuple{Float64,Float64,Float64,Bool}}(),
        Dict{X,Vector{Tuple{PricingSubproblemId,V}}}(),
        Dict{Tuple{PricingSubproblemId,V},Vector{X}}(),
        Set{X}(),
        Tuple{C,ConstraintSense,Float64}[]
    )
end

function add_subproblem!(
    b::DecompositionBuilder{V,X,C,Y},
    sp_id::PricingSubproblemId,
    fixed_cost::Float64,
    conv_lb::Float64,
    conv_ub::Float64
) where {V,X,C,Y}
    b.sp_variables[sp_id] = V[]
    b.sp_original_costs[sp_id] = Dict{V,Float64}()
    b.sp_coupling_entries[sp_id] = Tuple{V,C,Float64}[]
    b.sp_fixed_costs[sp_id] = fixed_cost
    b.sp_conv_bounds[sp_id] = (conv_lb, conv_ub)
    return nothing
end

function add_sp_variable!(
    b::DecompositionBuilder{V}, sp_id::PricingSubproblemId,
    sp_var::V, orig_cost::Float64
) where {V}
    push!(b.sp_variables[sp_id], sp_var)
    b.sp_original_costs[sp_id][sp_var] = orig_cost
    return nothing
end

function add_coupling_coefficient!(
    b::DecompositionBuilder{V,X,C}, sp_id::PricingSubproblemId,
    sp_var::V, cstr_id::C, coeff::Float64
) where {V,X,C}
    push!(get!(Vector{Tuple{V,C,Float64}}, b.sp_coupling_entries, sp_id), (sp_var, cstr_id, coeff))
    return nothing
end

function add_mapping!(
    b::DecompositionBuilder{V,X}, orig_var::X,
    sp_id::PricingSubproblemId, sp_var::V
) where {V,X}
    push!(get!(Vector{Tuple{PricingSubproblemId,V}}, b.forward_map, orig_var), (sp_id, sp_var))
    push!(get!(Vector{X}, b.inverse_map, (sp_id, sp_var)), orig_var)
    push!(b.all_orig_vars_set, orig_var)
    return nothing
end

function add_pure_master_variable!(
    b::DecompositionBuilder{V,X,C,Y},
    y_id::Y,
    cost::Float64,
    lb::Float64,
    ub::Float64,
    is_integer::Bool
) where {V,X,C,Y}
    b.pm_data_accum[y_id] = (cost, lb, ub, is_integer)
    b.pm_coupling_accum[y_id] = CouplingEntry[]
    return nothing
end

function add_pure_master_coupling!(
    b::DecompositionBuilder{V,X,C,Y}, y_id::Y, cstr_id::C,
    coeff::Float64
) where {V,X,C,Y}
    push!(b.pm_coupling_accum[y_id], CouplingEntry(TaggedCI(cstr_id), coeff))
    return nothing
end

function add_coupling_constraint!(
    b::DecompositionBuilder{V,X,C}, cstr_id::C,
    sense::ConstraintSense, rhs::Float64
) where {V,X,C}
    push!(b.coupling_cstrs, (cstr_id, sense, rhs))
    return nothing
end

"""
    build(builder) -> Decomposition

Compile the accumulated data into an immutable Decomposition.
Builds the CSR coupling coefficient structure in O(total entries).
"""
function build(b::DecompositionBuilder{V,X,C,Y}) where {V,X,C,Y}
    subproblems = Dict{PricingSubproblemId,SubproblemData{V}}()

    for sp_id in keys(b.sp_variables)
        variables = b.sp_variables[sp_id]
        original_costs = b.sp_original_costs[sp_id]

        entries = CouplingEntry[]
        offsets = Dict{V,UnitRange{Int}}()

        # Group raw entries by sp_var using a dict (avoids requiring isless on V)
        raw_entries = get(b.sp_coupling_entries, sp_id, Tuple{V,C,Float64}[])
        grouped = Dict{V,Vector{Tuple{C,Float64}}}()
        for (sp_var, cstr_id, coeff) in raw_entries
            grp = get!(Vector{Tuple{C,Float64}}, grouped, sp_var)
            push!(grp, (cstr_id, coeff))
        end

        # Build CSR in the same order as the variables list (deterministic).
        # Each variable's slice is sorted by TaggedCI to enable merge-based
        # reduced cost computation (two-pointer merge with sorted duals).
        for sp_var in variables
            haskey(grouped, sp_var) || continue
            range_start = length(entries) + 1
            for (cstr_id, coeff) in grouped[sp_var]
                push!(entries, CouplingEntry(TaggedCI(cstr_id), coeff))
            end
            # @view into Vector is mutable — sort! operates in-place
            sort!(@view(entries[range_start:end]); by = e -> e.constraint_id)
            offsets[sp_var] = range_start:length(entries)
        end

        coupling_coeffs = CouplingCoefficients(entries, offsets)
        conv_lb, conv_ub = b.sp_conv_bounds[sp_id]

        subproblems[sp_id] = SubproblemData(
            variables, original_costs, coupling_coeffs,
            b.sp_fixed_costs[sp_id], conv_lb, conv_ub
        )
    end

    pm_vars = PureMasterVariableData{Y}[]
    for (y_id, (cost, lb, ub, is_int)) in b.pm_data_accum
        coeffs = get(b.pm_coupling_accum, y_id, CouplingEntry[])
        push!(pm_vars, PureMasterVariableData(y_id, cost, lb, ub, is_int, coeffs))
    end

    mapping = ForwardMapping(
        b.forward_map,
        b.inverse_map,
        collect(b.all_orig_vars_set)
    )

    return Decomposition{V,X,C,Y}(
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
