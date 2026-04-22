# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    PrimalMoiSolution

Unified primal solution type for both master and pricing problems.

Fields:
- obj_value::Float64: Objective function value of the solution
- variable_values::Dict{MOI.VariableIndex,Float64}: Variable index to value mapping
"""
struct PrimalMoiSolution
    obj_value::Float64
    variable_values::Dict{MOI.VariableIndex,Float64}
end

"""
    DualMoiSolution

Unified dual solution type for both master and pricing problems.

Fields:
- obj_value::Float64: Dual objective function value
- constraint_duals::Dict{TaggedCI,Float64}: Constraint dual values keyed
  by `TaggedCI` (encodes MOI constraint type via `CIKind` and index value)
"""
struct DualMoiSolution
    obj_value::Float64
    constraint_duals::Dict{TaggedCI,Float64}
end

function Base.show(io::IO, sol::PrimalMoiSolution)
    println(io, "Primal solution:")
    sorted_vars = sort(collect(sol.variable_values), by = x -> x[1].value)
    for (i, (var_index, value)) in enumerate(sorted_vars)
        connector = i == length(sorted_vars) ? "└" : "|"
        println(io, "$connector _[$(var_index.value)]: $value")
    end
    print(io, "└ cost = $(sol.obj_value)")
end

function Base.show(io::IO, sol::DualMoiSolution)
    println(io, "Dual solution:")
    sorted = sort(collect(sol.constraint_duals); by = first)
    for (i, (tagged, dual_value)) in enumerate(sorted)
        connector = i == length(sorted) ? "└" : "|"
        println(io, "$connector constr[$(tagged)]: $dual_value")
    end
    print(io, "└ cost = $(sol.obj_value)")
end

