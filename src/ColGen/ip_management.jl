# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

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

# Vanderbeck (2009) IP check: iterate all column variables, verify integer multiplicities.
# Returns (MasterIpPrimalSol, false) if all values are integral, (nothing, false) if not.
function _project_if_integral(mast_primal_sol, ws; tol=INTEGRALITY_TOL)
    non_zero_integral = Tuple{MOI.VariableIndex,Int}[]
    non_zero_continuous = Tuple{MOI.VariableIndex,Float64}[]
    obj = 0.0
    for (col_var, rec) in columns(ws.pool)
        val = get(mast_primal_sol.sol.variable_values, col_var, 0.0)
        rounded = round(val)
        abs(val - rounded) > tol && return nothing, false
        ival = round(Int, rounded)
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
            rounded = round(val)
            abs(val - rounded) > tol && return nothing, false
            ival = round(Int, rounded)
            if ival != 0
                push!(non_zero_integral, (pmv.id, ival))
                obj += cost * ival
            end
        else
            if abs(val) > tol
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
    _has_artificial_vars_in_solution(ws, mast_primal_sol) && return nothing, false
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
