# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

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
- constraint_duals::Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}: Constraint dual
  values organized by constraint type
"""
struct DualMoiSolution
    obj_value::Float64
    constraint_duals::Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}
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
    all_constraints = []
    for (constraint_type, constraint_dict) in sol.constraint_duals
        for (index_value, dual_value) in constraint_dict
            push!(all_constraints, (constraint_type, index_value, dual_value))
        end
    end
    sort!(all_constraints, by = x -> (string(x[1]), x[2]))
    for (i, (constraint_type, index_value, dual_value)) in enumerate(all_constraints)
        connector = i == length(all_constraints) ? "└" : "|"
        println(io, "$connector constr[$(constraint_type)][$(index_value)]: $dual_value")
    end
    print(io, "└ cost = $(sol.obj_value)")
end

"""
    recompute_cost(dual_sol::DualMoiSolution, model)::Float64

Recompute the dual objective cost by multiplying dual values with RHS values.
"""
function recompute_cost(dual_sol::DualMoiSolution, model)::Float64
    total_cost = 0.0
    for (constraint_type, constraint_dict) in dual_sol.constraint_duals
        for (index_value, dual_value) in constraint_dict
            constraint_index = constraint_type(index_value)
            try
                constraint_set = MOI.get(model, MOI.ConstraintSet(), constraint_index)
                rhs_value = if constraint_set isa MOI.LessThan
                    constraint_set.upper
                elseif constraint_set isa MOI.GreaterThan
                    constraint_set.lower
                elseif constraint_set isa MOI.EqualTo
                    constraint_set.value
                else
                    continue
                end
                total_cost += dual_value * rhs_value
            catch
                continue
            end
        end
    end
    try
        objective_function = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        total_cost += objective_function.constant
    catch
    end
    return total_cost
end
