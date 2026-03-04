# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct ReducedCosts
    values::Dict{Any,Dict{Any,Float64}}
end

"""
    SortedDualVector

Sorted vector of (TaggedCI, Float64) pairs built once per CG iteration
from the master dual solution. Enables merge-based reduced cost
computation with sorted coupling entries.
"""
struct SortedDualVector
    entries::Vector{Tuple{TaggedCI,Float64}}
end

function _build_sorted_duals(mast_dual_sol::MasterDualSolution)
    cc_ids = mast_dual_sol.coupling_constraint_ids
    result = Vector{Tuple{TaggedCI,Float64}}(undef, length(cc_ids))
    for (i, tagged) in enumerate(cc_ids)
        result[i] = (tagged, _dual_value(mast_dual_sol, tagged))
    end
    sort!(result; by = first)
    return SortedDualVector(result)
end

"""
    _merge_dual_contrib(coupling_view, sorted_duals) -> Float64

Two-pointer merge of sorted coupling entries against sorted dual vector.
Both sides must be sorted by TaggedCI. Returns Σ aᵢⱼ·πᵢ.
"""
function _merge_dual_contrib(
    coupling_view::AbstractVector{CouplingEntry},
    sorted_duals::SortedDualVector
)
    contrib = 0.0
    duals = sorted_duals.entries
    j = 1
    m = length(duals)
    for entry in coupling_view
        cid = entry.constraint_id
        while j <= m && duals[j][1] < cid
            j += 1
        end
        if j <= m && duals[j][1] == cid
            contrib += entry.coefficient * duals[j][2]
        end
    end
    return contrib
end

"""
    _dual_value(dual_sol, cstr_idx) -> Float64

Extract dual value for a specific constraint index from a MasterDualSolution.
Returns 0.0 if the constraint type or index is not present.
"""
function _dual_value(dual_sol::MasterDualSolution, cstr_idx)
    d = get(dual_sol.sol.constraint_duals, typeof(cstr_idx), nothing)
    isnothing(d) && return 0.0
    return get(d, cstr_idx.value, 0.0)
end

function _dual_value(dual_sol::MasterDualSolution, idx::TaggedCI)
    d = get(dual_sol.sol.constraint_duals, _ci_type(idx.kind), nothing)
    isnothing(d) && return 0.0
    return get(d, idx.value, 0.0)
end

# Hot path: sorted duals built once, shared across subproblems.
function _compute_sp_reduced_costs(
    ctx::ColGenContext, mast_dual_sol::MasterDualSolution,
    sorted_duals::SortedDualVector, sp_id; zero_cost=false
)
    decomp = ctx.decomp
    sp_rc = Dict{Any,Float64}()
    for sp_var in subproblem_variables(decomp, sp_id)
        rc = zero_cost ? 0.0 : original_cost(decomp, sp_id, sp_var)
        rc -= _merge_dual_contrib(
            coupling_coefficients(decomp, sp_id, sp_var), sorted_duals
        )
        # TODO: O(n_cuts) per variable — needs batch/merge optimization.
        rc -= total_cut_dual_contribution(ctx.cuts, sp_id, sp_var)
        # TODO: O(n_bc × n_orig_per_var) per variable — needs optimization.
        for bc in ctx.branching_constraints
            σ = _dual_value(mast_dual_sol, bc.constraint_index)
            iszero(σ) && continue
            for ov in mapping_to_original(decomp, sp_id, sp_var)
                if ov == bc.orig_var
                    rc -= σ
                    break
                end
            end
        end
        sp_rc[sp_var] = rc
    end
    return sp_rc
end

# Fallback: builds sorted duals per call (used by dw_stabilization.jl).
function _compute_sp_reduced_costs(
    ctx::ColGenContext, mast_dual_sol::MasterDualSolution, sp_id;
    zero_cost=false
)
    sorted_duals = _build_sorted_duals(mast_dual_sol)
    return _compute_sp_reduced_costs(
        ctx, mast_dual_sol, sorted_duals, sp_id; zero_cost=zero_cost
    )
end

function compute_reduced_costs!(
    ctx::ColGenContext, ::Union{Phase0,Phase2},
    mast_dual_sol::MasterDualSolution
)
    sorted_duals = _build_sorted_duals(mast_dual_sol)
    return ReducedCosts(Dict(
        sp_id => _compute_sp_reduced_costs(
            ctx, mast_dual_sol, sorted_duals, sp_id
        )
        for sp_id in subproblem_ids(ctx.decomp)
    ))
end

function compute_reduced_costs!(
    ctx::ColGenContext, ::Phase1, mast_dual_sol::MasterDualSolution
)
    sorted_duals = _build_sorted_duals(mast_dual_sol)
    return ReducedCosts(Dict(
        sp_id => _compute_sp_reduced_costs(
            ctx, mast_dual_sol, sorted_duals, sp_id; zero_cost=true
        )
        for sp_id in subproblem_ids(ctx.decomp)
    ))
end

function update_reduced_costs!(ctx::ColGenContext, ::Union{Phase0,Phase1,Phase2}, red_costs::ReducedCosts)
    # For maximization, negate RC so the SP (which always minimizes) finds the
    # column with the most positive reduced cost (= most negative negated RC).
    sign = is_minimization(ctx) ? 1.0 : -1.0
    for (sp_id, sp_rc) in red_costs.values
        sp_model = ctx.sp_models[sp_id]
        for (var_index, rc_value) in sp_rc
            MOI.modify(
                sp_model,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarCoefficientChange(var_index, sign * rc_value)
            )
        end
    end
    return nothing
end

compute_sp_init_db(ctx::ColGenContext, _) = is_minimization(ctx) ? -Inf : Inf
compute_sp_init_pb(ctx::ColGenContext, _) = is_minimization(ctx) ? Inf : -Inf
