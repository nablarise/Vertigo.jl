# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    ColumnRecord

A column stored in the pool. `original_cost` is the column's cost in the
master's original objective.
"""
struct ColumnRecord
    sp_id::PricingSubproblemId
    solution::_SpSolution
    original_cost::Float64
end

column_sp_id(rec::ColumnRecord) = rec.sp_id
column_original_cost(rec::ColumnRecord) = rec.original_cost
pricing_objective_value(rec::ColumnRecord) = rec.solution.obj_value
column_nonzero_entries(rec::ColumnRecord) = rec.solution.entries

"""
    ColumnPool

Triple-indexed column pool: by column variable, by subproblem, and by
fingerprint for O(1) dedup.
"""
mutable struct ColumnPool <: AbstractColumnPool
    by_column_var::Dict{_VI,ColumnRecord}
    by_subproblem::Dict{PricingSubproblemId,Vector{_VI}}
    fingerprints::Dict{PricingSubproblemId,Set{UInt64}}
end

function ColumnPool()
    return ColumnPool(
        Dict{_VI,ColumnRecord}(),
        Dict{PricingSubproblemId,Vector{_VI}}(),
        Dict{PricingSubproblemId,Set{UInt64}}()
    )
end

"""
    record_column!(pool, col_var, sp_id, sol, original_cost)

Register a column in the pool, indexing it by column variable, subproblem,
and fingerprint.
"""
function record_column!(
    pool::ColumnPool, col_var::_VI, sp_id::PricingSubproblemId,
    sol::_SpSolution, original_cost::Float64
)
    pool.by_column_var[col_var] = ColumnRecord(sp_id, sol, original_cost)
    sp_cols = get!(Vector{_VI}, pool.by_subproblem, sp_id)
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
function columns_for_subproblem(pool::ColumnPool, sp_id::PricingSubproblemId)
    col_vars = get(pool.by_subproblem, sp_id, _VI[])
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
