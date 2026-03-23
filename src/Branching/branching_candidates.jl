# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    BranchingCandidate{X}

A fractional original variable eligible for branching.
"""
struct BranchingCandidate{X}
    orig_var::X
    value::Float64
    floor_val::Float64
    ceil_val::Float64
    fractionality::Float64
end

"""
    find_fractional_variables(ctx, primal_values; tol=1e-6)

Project master LP solution to original-variable space and return
all fractional variables as `BranchingCandidate`s, sorted by
fractionality descending (most fractional first).
"""
function find_fractional_variables(
    ctx,
    primal_values::Dict{MOI.VariableIndex,Float64};
    tol::Float64 = 1e-6
)
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)
    x_values = project_to_original(
        decomp, pool, v -> get(primal_values, v, 0.0)
    )

    candidates = BranchingCandidate[]
    for (orig_var, x_val) in x_values
        frac_part = x_val - floor(x_val)
        (frac_part < tol || frac_part > 1.0 - tol) && continue
        fractionality = min(frac_part, 1.0 - frac_part)
        push!(candidates, BranchingCandidate(
            orig_var, x_val, floor(x_val), ceil(x_val),
            fractionality
        ))
    end
    sort!(candidates; by=c -> c.fractionality, rev=true)
    return candidates
end
