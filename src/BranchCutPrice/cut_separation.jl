# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    _build_cut_saf(decomp, pool, master, cut)

Build a `MOI.ScalarAffineFunction` for a robust cut over the
columns currently in `pool`. Each column's coefficient is the
inner product of the cut's original-variable coefficients with
the column's subproblem solution.
"""
function _build_cut_saf(decomp, pool, master, cut)
    terms = MOI.ScalarAffineTerm{Float64}[]
    for (col_var, rec) in columns(pool)
        MOI.is_valid(master, col_var) || continue
        sp_id = column_sp_id(rec)
        coeff = 0.0
        for (sp_var, val) in column_nonzero_entries(rec)
            ov = mapped_original_var(decomp, sp_id, sp_var)
            ov === nothing && continue
            c = get(cut.coefficients, ov, 0.0)
            if !iszero(c)
                coeff += c * val
            end
        end
        if !iszero(coeff)
            push!(terms, MOI.ScalarAffineTerm(coeff, col_var))
        end
    end
    return MOI.ScalarAffineFunction(terms, 0.0)
end

"""
    _add_robust_cut!(space, separated_cut)

Add a single separated robust cut to the master model and register
it in the CG context.
"""
function _add_robust_cut!(space::BranchCutPriceWorkspace, cut::SeparatedCut)
    ws = space.ws
    saf = _build_cut_saf(
        bp_decomp(ws), bp_pool(ws), space.backend, cut
    )
    ci = MOI.add_constraint(space.backend, saf, cut.set)
    tagged = TaggedCI(ci)
    push!(
        bp_robust_cuts(ws),
        ColGen.ActiveRobustCut(tagged, cut.coefficients)
    )
    return tagged
end

"""
    _separate_and_add_cuts!(space, cg_output) -> Int

Project the master LP solution to original space, separate cuts,
and add them to the master. Returns the number of cuts added.
"""
function _separate_and_add_cuts!(
    space::BranchCutPriceWorkspace, cg_output::ColGen.ColGenOutput
)::Int
    isnothing(space.separator) && return 0
    cg_output.status != ColGen.optimal && return 0
    decomp = bp_decomp(space.ws)
    pool = bp_pool(space.ws)
    primal_cache = get_primal_solution(space.backend)
    x = project_to_original(
        decomp, pool,
        v -> get(primal_cache, v, 0.0)
    )
    cuts = separate(space.separator, x)
    for cut in cuts
        _add_robust_cut!(space, cut)
    end
    space.total_cuts_separated += length(cuts)
    return length(cuts)
end
