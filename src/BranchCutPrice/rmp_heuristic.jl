# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    solve_restricted_master_ip!(space, cg_output)

Solve the restricted master problem as a MIP to find IP-feasible
solutions from fractional LP relaxations. Adds integrality
constraints to column variables and integer pure master variables,
solves, then restores the LP relaxation for branching.

Returns a `ColGen.MasterIpPrimalSol` if a feasible IP solution
is found, `nothing` otherwise.
"""
function solve_restricted_master_ip!(
    space::BPSpace,
    cg_output::ColGen.ColGenOutput
)
    if cg_output.status == ColGen.master_infeasible ||
       cg_output.status == ColGen.subproblem_infeasible ||
       cg_output.status == ColGen.ip_pruned
        return nothing
    end

    backend = space.backend
    pool = bp_pool(space.ctx)
    decomp = bp_decomp(space.ctx)

    # Add integrality constraints to column variables and integer
    # pure master variables.
    int_cis = MOI.ConstraintIndex{MOI.VariableIndex,MOI.Integer}[]
    for (col_var, _) in columns(pool)
        ci = MOI.add_constraint(backend, col_var, MOI.Integer())
        push!(int_cis, ci)
    end
    for pmv in pure_master_variables(decomp)
        if pure_master_is_integer(decomp, pmv)
            ci = MOI.add_constraint(
                backend, pmv.id, MOI.Integer()
            )
            push!(int_cis, ci)
        end
    end

    isempty(int_cis) && return nothing

    # Set time limit and solve MIP.
    MOI.set(backend, MOI.TimeLimitSec(), space.rmp_time_limit)
    MOI.optimize!(backend)

    result = nothing
    if MOI.get(backend, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        obj = MOI.get(backend, MOI.ObjectiveValue())
        nz_int = Tuple{MOI.VariableIndex,Int}[]
        nz_cont = Tuple{MOI.VariableIndex,Float64}[]
        for (col_var, _) in columns(pool)
            val = MOI.get(backend, MOI.VariablePrimal(), col_var)
            if val > 0.5
                push!(nz_int, (col_var, round(Int, val)))
            end
        end
        for pmv in pure_master_variables(decomp)
            val = MOI.get(
                backend, MOI.VariablePrimal(), pmv.id
            )
            if pure_master_is_integer(decomp, pmv)
                if abs(val) > 0.5
                    push!(nz_int, (pmv.id, round(Int, val)))
                end
            else
                if abs(val) > 1e-6
                    push!(nz_cont, (pmv.id, val))
                end
            end
        end
        if !isempty(nz_int) || !isempty(nz_cont)
            result = ColGen.MasterIpPrimalSol(
                obj, nz_int, nz_cont
            )
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
