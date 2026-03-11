# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    _compute_robust_cut_column_coeff(decomp, coefficients, sp_id, sol)

Compute a single column's coefficient in a robust cut defined by
`coefficients` (orig_var => coeff).
"""
function _compute_robust_cut_column_coeff(
    decomp, coefficients::Dict{Any,Float64}, sp_id, sol
)
    coeff = 0.0
    for (sp_var, val) in nonzero_entries(sol)
        for ov in mapping_to_original(decomp, sp_id, sp_var)
            c = get(coefficients, ov, 0.0)
            if !iszero(c)
                coeff += c * val
                break
            end
        end
    end
    return coeff
end

"""
    _add_robust_cut!(space, separated_cut)

Add a single separated robust cut to the master model and register
it in the CG context.
"""
function _add_robust_cut!(space::BPSpace, cut::SeparatedCut)
    ctx = space.ctx
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)
    master = space.backend

    # Build SAF terms from existing columns
    terms = MOI.ScalarAffineTerm{Float64}[]
    for (col_var, rec) in columns(pool)
        MOI.is_valid(master, col_var) || continue
        sp_id = column_sp_id(rec)
        sol_entries = column_nonzero_entries(rec)
        coeff = 0.0
        for (sp_var, val) in sol_entries
            for ov in mapping_to_original(decomp, sp_id, sp_var)
                c = get(cut.coefficients, ov, 0.0)
                if !iszero(c)
                    coeff += c * val
                    break
                end
            end
        end
        if !iszero(coeff)
            push!(terms, MOI.ScalarAffineTerm(coeff, col_var))
        end
    end

    saf = MOI.ScalarAffineFunction(terms, 0.0)
    ci = MOI.add_constraint(master, saf, cut.set)
    tagged = TaggedCI(ci)
    active = ColGen.ActiveRobustCut(tagged, cut.coefficients)
    push!(bp_robust_cuts(ctx), active)
    return tagged
end

"""
    _run_cut_separation_loop!(space, node)

Run the cut separation loop after CG converges at a node.
Returns the final `ColGenOutput` if cuts were added, or `nothing`.
"""
function _run_cut_separation_loop!(space::BPSpace, node)
    isnothing(space.separator) && return nothing
    space.max_cut_rounds <= 0 && return nothing

    ctx = space.ctx
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)
    master = space.backend
    final_output = nothing

    for _ in 1:space.max_cut_rounds
        # Project master LP solution to original space
        x = project_to_original(
            decomp, pool,
            v -> MOI.get(master, MOI.VariablePrimal(), v)
        )

        # Separate cuts
        cuts = separate(space.separator, x)
        isempty(cuts) && break

        # Add each cut
        for cut in cuts
            _add_robust_cut!(space, cut)
        end

        # Re-run CG
        raw_ctx = ctx isa ColGen.ColGenLoggerContext ?
            ctx.inner : ctx
        raw_ctx.ip_primal_bound = isnothing(space.incumbent) ?
            nothing : space.incumbent.obj_value
        _rebuild_branching_constraints!(space)
        cg_output = ColGen.run_column_generation(ctx)
        node.user_data = BPNodeData(cg_output)
        final_output = cg_output

        # Stop if CG did not converge to optimal
        cg_output.status != ColGen.optimal && break
    end

    return final_output
end
