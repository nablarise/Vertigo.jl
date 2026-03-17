# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    get_primal_solution(model) -> Dict{MOI.VariableIndex,Float64}

Extract the primal variable values from a solved MOI model.
Returns an empty dictionary when the primal status is not
`MOI.FEASIBLE_POINT`.
"""
function get_primal_solution(model)
    result = Dict{MOI.VariableIndex,Float64}()
    primal_status = MOI.get(model, MOI.PrimalStatus())
    if primal_status == MOI.FEASIBLE_POINT
        for var in MOI.get(model, MOI.ListOfVariableIndices())
            result[var] = MOI.get(
                model, MOI.VariablePrimal(), var
            )
        end
    end
    return result
end
