# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

abstract type AbstractDecomposition end
abstract type AbstractSubproblemSolution end
abstract type AbstractColumnPool end

# ── Variable key type ────────────────────────────────────────────────────────
function orig_var_type end

# ── M⁻¹ direction (z → x): hot path ──────────────────────────────────────────
function original_cost end
function coupling_coefficients end

# ── M direction (x → z): for propagation ────────────────────────────────────
function mapped_original_var end
function mapped_subproblem_var end
function original_variables end

# ── Subproblem queries ────────────────────────────────────────────────────────
function subproblem_ids end
function subproblem_variables end
function subproblem_fixed_cost end
function convexity_bounds end
function nb_subproblem_multiplicity end

# ── Pure master variable queries ──────────────────────────────────────────────
function pure_master_variables end
function pure_master_cost end
function pure_master_bounds end
function pure_master_is_integer end
function pure_master_coupling_coefficients end

# ── Master constraints and sense ──────────────────────────────────────────────
function coupling_constraints end
function is_minimization end

# ── Subproblem solution interface ────────────────────────────────────────────
function solution_value end
function nonzero_entries end
function subproblem_id end
function objective_value end

# ── Column pool interface ─────────────────────────────────────────────────────
function record_column! end
function get_column_solution end
function get_column_sp_id end
function get_column_cost end
function columns end
function columns_for_subproblem end
function has_column end

# ── Bound propagation ─────────────────────────────────────────────────────────
function propagate_bounds! end
function is_column_proper end

# ────────────────────────────────────────────────────────────────────────────────────────
# DERIVED COMPUTATIONS (generic — use the interface above)
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    compute_column_original_cost(decomp, sp_id, sol) -> Float64

Original cost of a column: fₖ + Σ_{(z,v) ∈ sol} original_cost(z) · v
"""
function compute_column_original_cost(
    decomp::AbstractDecomposition, sp_id, sol::AbstractSubproblemSolution
)
    cost = subproblem_fixed_cost(decomp, sp_id)
    for (sp_var, val) in nonzero_entries(sol)
        cost += original_cost(decomp, sp_id, sp_var) * val
    end
    return cost
end

"""
    compute_column_coupling_coefficients(decomp, sp_id, sol) -> Dict{Any, Float64}

Coupling constraint coefficients of a column:
    coeff_i(λₚ) = Σ_{(z,v) ∈ sol} coupling_coeff(z, i) · v
"""
function compute_column_coupling_coefficients(
    decomp::AbstractDecomposition, sp_id, sol::AbstractSubproblemSolution
)
    coeffs = Dict{Any,Float64}()
    for (sp_var, val) in nonzero_entries(sol)
        for entry in coupling_coefficients(decomp, sp_id, sp_var)
            cstr = entry.constraint_id
            coeffs[cstr] = get(coeffs, cstr, 0.0) + entry.coefficient * val
        end
    end
    return coeffs
end

"""
    compute_sp_reduced_costs(decomp, sp_id, dual_values) -> Dict{Any, Float64}

Reduced costs for all variables in subproblem sp_id:
    c̄ₜᵏ = original_cost(zₜᵏ) - Σᵢ coupling_coeff(zₜᵏ, i) · πᵢ

dual_values must support: dual_values(constraint_id) -> Float64
"""
function compute_sp_reduced_costs(decomp::AbstractDecomposition, sp_id, dual_values)
    red_costs = Dict{Any,Float64}()
    for sp_var in subproblem_variables(decomp, sp_id)
        rc = original_cost(decomp, sp_id, sp_var)
        for entry in coupling_coefficients(decomp, sp_id, sp_var)
            rc -= entry.coefficient * dual_values(entry.constraint_id)
        end
        red_costs[sp_var] = rc
    end
    return red_costs
end

"""
    compute_column_reduced_cost(decomp, sp_id, sol, sp_reduced_costs, convexity_duals) -> Float64

True reduced cost of a column:
    c̄(z̄ᵖ) = fₖ + Σₜ c̄ₜᵏ · z̄ₜᵖ - ν⁺ₖ - ν⁻ₖ
"""
function compute_column_reduced_cost(
    decomp::AbstractDecomposition,
    sp_id,
    sol::AbstractSubproblemSolution,
    sp_reduced_costs,
    convexity_duals::Tuple{Float64,Float64}
)
    rc = subproblem_fixed_cost(decomp, sp_id)
    for (sp_var, val) in nonzero_entries(sol)
        rc += sp_reduced_costs[sp_var] * val
    end
    ν_lb, ν_ub = convexity_duals
    rc -= ν_lb + ν_ub
    return rc
end

"""
    compute_dual_bound_pure_master_contribution(decomp, dual_values) -> Float64

Lagrangian contribution from pure master variables:
    Σₛ min{ (gₛ - Σᵢ φᵢₛ πᵢ) · Yₛ,  (gₛ - Σᵢ φᵢₛ πᵢ) · Ȳₛ }
"""
function compute_dual_bound_pure_master_contribution(decomp::AbstractDecomposition, dual_values)
    contrib = 0.0
    for y_var in pure_master_variables(decomp)
        rc_y = pure_master_cost(decomp, y_var)
        for entry in pure_master_coupling_coefficients(decomp, y_var)
            rc_y -= entry.coefficient * dual_values(entry.constraint_id)
        end
        lb, ub = pure_master_bounds(decomp, y_var)
        contrib += min(rc_y * lb, rc_y * ub)
    end
    return contrib
end

"""
    compute_branching_column_coefficient(decomp, orig_var, sp_id, sol) -> Float64

Coefficient of a column λ in a branching constraint on original variable `orig_var`.
Equals Σ_{(z,v) ∈ sol} v · [z maps to orig_var].
"""
function compute_branching_column_coefficient(
    decomp::AbstractDecomposition, orig_var, sp_id,
    sol::AbstractSubproblemSolution
)
    coeff = 0.0
    for (sp_var, val) in nonzero_entries(sol)
        if mapped_original_var(decomp, sp_id, sp_var) == orig_var
            coeff += val
        end
    end
    return coeff
end

"""
    project_to_original(decomp, pool, master_primal_values) -> Dict

Project master LP solution (λ̄, ȳ) back to original variable space x̄.
"""
function project_to_original(
    decomp::AbstractDecomposition, pool::AbstractColumnPool,
    master_primal_values
)
    x_values = Dict{Any,Float64}()
    for (col_var, rec) in columns(pool)
        λ_val = master_primal_values(col_var)
        iszero(λ_val) && continue
        sp_id = column_sp_id(rec)
        for (sp_var, z_val) in column_nonzero_entries(rec)
            orig_var = mapped_original_var(decomp, sp_id, sp_var)
            if orig_var !== nothing
                x_values[orig_var] = get(x_values, orig_var, 0.0) + z_val * λ_val
            end
        end
    end
    return x_values
end
