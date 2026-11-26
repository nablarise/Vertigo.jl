# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ────────────────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────────────────

# Check art vars using already-captured variable_values — avoids querying the MOI model.
function _has_artificial_vars_in_solution(ws, mast_primal_sol; tol=INTEGRALITY_TOL)
    vars = mast_primal_sol.sol.variable_values
    for (_, (s_pos, s_neg)) in ws.eq_art_vars
        (abs(get(vars, s_pos, 0.0)) > tol || abs(get(vars, s_neg, 0.0)) > tol) && return true
    end
    for (_, s) in ws.leq_art_vars
        abs(get(vars, s, 0.0)) > tol && return true
    end
    for (_, s) in ws.geq_art_vars
        abs(get(vars, s, 0.0)) > tol && return true
    end
    return false
end

"""
    is_master_integral(mast_primal_sol, ws::ColGenWorkspace) -> Bool

Whether the master solution is integral in the master itself: every column
value and every integer pure master variable is integral, and no artificial
variable is nonzero.

This is sufficient, not necessary. For a subproblem with upper multiplicity
1 whose Σλ is 0 or 1, an integral projection onto the original variables is
already a feasible subproblem solution, even when the columns behind it are
fractional. Only with upper multiplicity > 1 can fractional columns hide
behind an integral projection that no integral master solution reproduces.
"""
function is_master_integral(mast_primal_sol, ws::ColGenWorkspace)
    if _has_artificial_vars_in_solution(ws, mast_primal_sol)
        return false
    end
    vars = mast_primal_sol.sol.variable_values
    for (col_var, _) in columns(ws.pool)
        if !_is_integral(get(vars, col_var, 0.0))
            return false
        end
    end
    decomp = ws.decomp
    for pmv in pure_master_variables(decomp)
        if pure_master_is_integer(decomp, pmv) &&
                !_is_integral(get(vars, pmv.id, 0.0))
            return false
        end
    end
    return true
end

_is_integral(val) = abs(val - round(val)) <= INTEGRALITY_TOL

# Vanderbeck (2009) IP check: iterate all column variables, verify integer multiplicities.
# Returns (MasterIpPrimalSol, false) if the master solution is integral, (nothing, false) if not.
function _project_if_integral(mast_primal_sol, ws)
    if !is_master_integral(mast_primal_sol, ws)
        return nothing, false
    end
    non_zero_integral = Tuple{MOI.VariableIndex,Int}[]
    non_zero_continuous = Tuple{MOI.VariableIndex,Float64}[]
    # The master objective carries the constant of the original objective;
    # no phase changes it.
    obj = MOI.get(
        master_model(ws.decomp),
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
    ).constant
    for (col_var, rec) in columns(ws.pool)
        val = get(mast_primal_sol.sol.variable_values, col_var, 0.0)
        ival = round(Int, val)
        if ival > 0
            push!(non_zero_integral, (col_var, ival))
            obj += column_original_cost(rec) * ival
        end
    end
    decomp = ws.decomp
    for pmv in pure_master_variables(decomp)
        val = get(mast_primal_sol.sol.variable_values, pmv.id, 0.0)
        cost = pure_master_cost(decomp, pmv)
        if pure_master_is_integer(decomp, pmv)
            ival = round(Int, val)
            if ival != 0
                push!(non_zero_integral, (pmv.id, ival))
                obj += cost * ival
            end
        else
            if abs(val) > INTEGRALITY_TOL
                push!(non_zero_continuous, (pmv.id, val))
                obj += cost * val
            end
        end
    end
    return MasterIpPrimalSol(obj, non_zero_integral, non_zero_continuous),
           false
end

# ────────────────────────────────────────────────────────────────────────────────────────
# COLUNA INTERFACE
# ────────────────────────────────────────────────────────────────────────────────────────

function check_primal_ip_feasibility!(
    mast_primal_sol::MasterPrimalSolution,
    ws::ColGenWorkspace,
    ::CGPhase
)
    return _project_if_integral(mast_primal_sol, ws)
end

function _is_strictly_better(
    ws::ColGenWorkspace,
    candidate::MasterIpPrimalSol,
    incumbent::MasterIpPrimalSol
)
    is_minimization(ws) ? candidate.obj_value < incumbent.obj_value - RC_IMPROVING_TOL :
                           candidate.obj_value > incumbent.obj_value + RC_IMPROVING_TOL
end

function update_inc_primal_sol!(
    ws::ColGenWorkspace, ::Nothing, new_sol::MasterIpPrimalSol
)
    current = ws.ip_incumbent
    if isnothing(current) || _is_strictly_better(ws, new_sol, current)
        ws.ip_incumbent = new_sol
    end
    return nothing
end
