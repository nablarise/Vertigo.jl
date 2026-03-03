# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    solve_restricted_master_ip!(space, cg_output)

Solve the restricted master problem as a MIP to find IP-feasible
solutions from fractional LP relaxations. Adds integrality
constraints to column variables, solves, then restores the LP
relaxation for branching.

Returns a `ColGen.ProjectedIpPrimalSol` if a feasible IP solution
is found, `nothing` otherwise.
"""
function solve_restricted_master_ip!(
    space::BPSpace,
    cg_output::ColGen.ColGenOutput
)
    if cg_output.status == ColGen.master_infeasible ||
       cg_output.status == ColGen.subproblem_infeasible
        return nothing
    end

    backend = space.backend
    pool = bp_pool(space.ctx)

    # Add integrality constraints to column variables.
    int_cis = MOI.ConstraintIndex{MOI.VariableIndex,MOI.Integer}[]
    for (master_var, _, _, _) in ColGen.columns(pool)
        ci = MOI.add_constraint(backend, master_var, MOI.Integer())
        push!(int_cis, ci)
    end

    isempty(int_cis) && return nothing

    # Set time limit and solve MIP.
    MOI.set(backend, MOI.TimeLimitSec(), space.rmp_time_limit)
    MOI.optimize!(backend)

    result = nothing
    if MOI.get(backend, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        obj = MOI.get(backend, MOI.ObjectiveValue())
        selected = Tuple{MOI.VariableIndex,Int}[]
        for (master_var, _, _, _) in ColGen.columns(pool)
            val = MOI.get(backend, MOI.VariablePrimal(), master_var)
            if val > 0.5
                push!(selected, (master_var, round(Int, val)))
            end
        end
        if !isempty(selected)
            result = ColGen.ProjectedIpPrimalSol(obj, selected)
        end
    end

    # Remove integrality constraints to restore LP relaxation.
    for ci in int_cis
        MOI.delete(backend, ci)
    end

    # Clear time limit and re-solve LP for branching.
    MOI.set(backend, MOI.TimeLimitSec(), nothing)
    MOI.optimize!(backend)

    return result
end
