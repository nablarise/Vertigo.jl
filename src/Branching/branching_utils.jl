# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    most_fractional_original_variable(ctx, primal_values; tol=1e-6)

Project the master LP solution to original-variable space and find the
most fractional original variable (closest to 0.5). Returns
`(orig_var, x_val)` or `(nothing, nothing)` if all are integral.
"""
function most_fractional_original_variable(
    ctx,
    primal_values::Dict{MOI.VariableIndex,Float64};
    tol::Float64 = 1e-6
)
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)
    x_values = project_to_original(
        decomp, pool, v -> get(primal_values, v, 0.0)
    )

    best_var = nothing
    best_val = 0.0
    best_dist = 1.0

    for (orig_var, x_val) in x_values
        frac_part = x_val - floor(x_val)
        (frac_part < tol || frac_part > 1.0 - tol) && continue
        dist = abs(frac_part - 0.5)
        if dist < best_dist
            best_dist = dist
            best_var = orig_var
            best_val = x_val
        end
    end
    return best_var, best_val
end
