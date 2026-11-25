# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# Recompute the objective of a restricted master IP solution from the master
# data and throw if the value the solver reported differs. Falling back to the
# recomputed value would hide a solver that returns inconsistent results.
# `var_values` maps master variables to their value; a missing variable is 0.
function _check_rmp_ip_objective(decomp, pool, var_values, reported; rtol=1e-6)
    # The master objective carries the constant of the original objective.
    recomputed = MOI.get(
        master_model(decomp),
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
    ).constant
    for (col_var, rec) in columns(pool)
        recomputed += column_original_cost(rec) * get(var_values, col_var, 0.0)
    end
    for pmv in pure_master_variables(decomp)
        recomputed += pure_master_cost(decomp, pmv) * get(var_values, pmv.id, 0.0)
    end
    if !isapprox(reported, recomputed; rtol=rtol, atol=rtol)
        error(
            "Restricted master IP objective mismatch: solver reported " *
            "$reported, solution costs $recomputed"
        )
    end
    return nothing
end

"""
    solve_restricted_master_ip!(space, cg_output)

Solve the restricted master problem as a MIP to find IP-feasible
solutions from fractional LP relaxations. Adds integrality
constraints to column variables and integer pure master variables,
solves, then removes them.

Returns a `ColGen.MasterIpPrimalSol` if a feasible IP solution
is found, `nothing` otherwise.
"""
function solve_restricted_master_ip!(
    space::BranchCutPriceWorkspace,
    cg_output::ColGen.ColGenOutput
)
    if cg_output.status == ColGen.master_infeasible ||
       cg_output.status == ColGen.subproblem_infeasible ||
       cg_output.status == ColGen.ip_pruned
        return nothing
    end

    backend = space.backend
    pool = bp_pool(space.ws)
    decomp = bp_decomp(space.ws)

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
    time_left = max(space.deadline - time(), 0.0)
    MOI.set(
        backend, MOI.TimeLimitSec(), min(space.rmp_time_limit, time_left)
    )
    MOI.optimize!(backend)

    # A solution that uses an artificial variable does not cover the master
    # constraints with columns only: it is not an IP-feasible solution.
    result = nothing
    if MOI.get(backend, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT &&
       !ColGen.has_artificial_vars_in_solution(space.ws)
        obj = MOI.get(backend, MOI.ObjectiveValue())
        var_values = Dict{MOI.VariableIndex,Float64}()
        nz_int = Tuple{MOI.VariableIndex,Int}[]
        nz_cont = Tuple{MOI.VariableIndex,Float64}[]
        for (col_var, _) in columns(pool)
            val = MOI.get(backend, MOI.VariablePrimal(), col_var)
            var_values[col_var] = val
            if val > 0.5
                push!(nz_int, (col_var, round(Int, val)))
            end
        end
        for pmv in pure_master_variables(decomp)
            val = MOI.get(
                backend, MOI.VariablePrimal(), pmv.id
            )
            var_values[pmv.id] = val
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
        _check_rmp_ip_objective(decomp, pool, var_values, obj)
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

    MOI.set(backend, MOI.TimeLimitSec(), nothing)

    return result
end
