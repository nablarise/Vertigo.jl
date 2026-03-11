# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

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
    _separate_and_add_cuts!(space, cg_output) -> Int

Project the master LP solution to original space, separate cuts,
and add them to the master. Returns the number of cuts added.
"""
function _separate_and_add_cuts!(
    space::BPSpace, cg_output::ColGen.ColGenOutput
)::Int
    isnothing(space.separator) && return 0
    cg_output.status != ColGen.optimal && return 0
    decomp = bp_decomp(space.ctx)
    pool = bp_pool(space.ctx)
    x = project_to_original(
        decomp, pool,
        v -> MOI.get(space.backend, MOI.VariablePrimal(), v)
    )
    cuts = separate(space.separator, x)
    for cut in cuts
        _add_robust_cut!(space, cut)
    end
    return length(cuts)
end
