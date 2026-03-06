# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    _SpSolution

Concrete subproblem solution with sorted entries for deterministic order and
cheap deduplication via fingerprint hash.

`obj_value` is the pricing subproblem objective (reduced-cost objective).
"""
struct _SpSolution <: AbstractSubproblemSolution
    sp_id::PricingSubproblemId
    obj_value::Float64
    entries::Vector{Tuple{_VI,Float64}}  # sorted by sp_var
    fingerprint::UInt64
end

function _SpSolution(sp_id::PricingSubproblemId, obj_value::Float64, entries::Vector{Tuple{_VI,Float64}})
    sorted = sort(entries; by = e -> e[1].value)
    filter!(e -> !iszero(e[2]), sorted)
    fp = hash(map(e -> (e[1].value, round(e[2]; digits=10)), sorted))
    return _SpSolution(sp_id, obj_value, sorted, fp)
end

subproblem_id(sol::_SpSolution) = sol.sp_id
objective_value(sol::_SpSolution) = sol.obj_value
@inline nonzero_entries(sol::_SpSolution) = sol.entries

function solution_value(sol::_SpSolution, sp_var::_VI)
    idx = searchsortedfirst(sol.entries, (sp_var, -Inf); by=first)
    if idx <= length(sol.entries) && first(sol.entries[idx]) == sp_var
        return sol.entries[idx][2]
    end
    return 0.0
end
