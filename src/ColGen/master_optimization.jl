# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct MasterPrimalSolution
    sol::PrimalMoiSolution
end

struct MasterDualSolution
    sol::DualMoiSolution
    # Coupling constraint IDs used to filter the full dual solution and
    # build a sorted dual vector for merge-based reduced cost computation.
    coupling_constraint_ids::Vector{TaggedCI}
end

Base.show(io::IO, sol::MasterPrimalSolution) = show(io, sol.sol)
Base.show(io::IO, sol::MasterDualSolution) = show(io, sol.sol)


struct MasterSolution
    moi_termination_status::MOI.TerminationStatusCode
    moi_primal_status::MOI.ResultStatusCode
    moi_dual_status::MOI.ResultStatusCode
    primal_sol::MasterPrimalSolution
    dual_sol::MasterDualSolution
end

is_infeasible(sol::MasterSolution) = sol.moi_termination_status == MOI.INFEASIBLE
function is_unbounded(sol::MasterSolution)
    return sol.moi_termination_status == MOI.DUAL_INFEASIBLE ||
           sol.moi_termination_status == MOI.INFEASIBLE_OR_UNBOUNDED
end
get_obj_val(sol::MasterSolution) = sol.primal_sol.sol.obj_value
get_primal_sol(sol::MasterSolution) = sol.primal_sol
get_dual_sol(sol::MasterSolution) = sol.dual_sol

is_better_primal_sol(::MasterPrimalSolution, ::Nothing) = true

function _populate_variable_values(model)
    variable_values = Dict{MOI.VariableIndex,Float64}()
    primal_status = MOI.get(model, MOI.PrimalStatus())
    if primal_status == MOI.FEASIBLE_POINT
        variables = MOI.get(model, MOI.ListOfVariableIndices())
        for var in variables
            variable_values[var] = MOI.get(model, MOI.VariablePrimal(), var)
        end
    end
    return variable_values
end

function _populate_constraint_duals(model)
    constraint_duals = Dict{TaggedCI,Float64}()
    dual_status = MOI.get(model, MOI.DualStatus())
    dual_status == MOI.FEASIBLE_POINT || return constraint_duals
    sense = MOI.get(model, MOI.ObjectiveSense()) == MOI.MAX_SENSE ? -1 : 1
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
            dual_val = MOI.get(model, MOI.ConstraintDual(), ci)
            constraint_duals[TaggedCI(ci)] = sense * dual_val
        end
    end
    return constraint_duals
end

function optimize_master_lp_problem!(master, ::ColGenContext)
    MOI.optimize!(moi_master(master))

    obj_value = MOI.get(moi_master(master), MOI.ObjectiveValue())
    variable_values = _populate_variable_values(moi_master(master))
    primal_sol = MasterPrimalSolution(PrimalMoiSolution(obj_value, variable_values))

    dual_obj_value = MOI.get(moi_master(master), MOI.DualObjectiveValue())
    constraint_duals = _populate_constraint_duals(moi_master(master))
    dual_sol = MasterDualSolution(
        DualMoiSolution(dual_obj_value, constraint_duals),
        master.coupling_constraint_ids
    )

    return MasterSolution(
        MOI.get(moi_master(master), MOI.TerminationStatus()),
        MOI.get(moi_master(master), MOI.PrimalStatus()),
        MOI.get(moi_master(master), MOI.DualStatus()),
        primal_sol,
        dual_sol
    )
end

function update_master_constrs_dual_vals!(::ColGenContext, ::MasterDualSolution)
    return nothing
end
