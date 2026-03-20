# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

const ZERO_TOL = 1e-8

"""
    get_primal_solution(model) -> Dict{MOI.VariableIndex,Float64}

Extract the primal variable values from a solved MOI model.
Returns a sparse dictionary — entries with `abs(val) <= ZERO_TOL`
are skipped.

The caller is responsible for ensuring the model has a valid primal
solution before calling this function.
"""
function get_primal_solution(model)
    result = Dict{MOI.VariableIndex,Float64}()
    for var in MOI.get(model, MOI.ListOfVariableIndices())
        val = MOI.get(model, MOI.VariablePrimal(), var)
        if abs(val) > ZERO_TOL
            result[var] = val
        end
    end
    return result
end

"""
    get_dual_solution(model) -> Dict{TaggedCI,Float64}

Extract constraint dual values from a solved MOI model.
Returns a sparse dictionary — entries with `abs(val) <= ZERO_TOL`
are skipped. Duals are sign-normalized so that positive values
always mean "tightening the constraint increases the objective"
regardless of the optimization sense.

The caller is responsible for ensuring the model has a valid dual
solution before calling this function.
"""
function get_dual_solution(model)
    result = Dict{TaggedCI,Float64}()
    sense = MOI.get(
        model, MOI.ObjectiveSense()
    ) == MOI.MAX_SENSE ? -1 : 1
    for (F, S) in MOI.get(
        model, MOI.ListOfConstraintTypesPresent()
    )
        for ci in MOI.get(
            model, MOI.ListOfConstraintIndices{F,S}()
        )
            dual_val = sense * MOI.get(
                model, MOI.ConstraintDual(), ci
            )
            if abs(dual_val) > ZERO_TOL
                result[TaggedCI(ci)] = dual_val
            end
        end
    end
    return result
end
