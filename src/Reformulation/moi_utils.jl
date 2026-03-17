# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

const ZERO_TOL = 1e-8

"""
    get_primal_solution(model) -> Dict{MOI.VariableIndex,Float64}

Extract the primal variable values from a solved MOI model.
Returns a sparse dictionary — entries with `abs(val) <= ZERO_TOL`
are skipped. Returns an empty dictionary when the primal status
is not `MOI.FEASIBLE_POINT`.
"""
function get_primal_solution(model)
    result = Dict{MOI.VariableIndex,Float64}()
    primal_status = MOI.get(model, MOI.PrimalStatus())
    if primal_status == MOI.FEASIBLE_POINT
        for var in MOI.get(model, MOI.ListOfVariableIndices())
            val = MOI.get(model, MOI.VariablePrimal(), var)
            if abs(val) > ZERO_TOL
                result[var] = val
            end
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
regardless of the optimization sense. Returns an empty dictionary
when the dual status is not `MOI.FEASIBLE_POINT`.
"""
function get_dual_solution(model)
    result = Dict{TaggedCI,Float64}()
    dual_status = MOI.get(model, MOI.DualStatus())
    if dual_status != MOI.FEASIBLE_POINT
        return result
    end
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
