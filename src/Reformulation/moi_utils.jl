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
