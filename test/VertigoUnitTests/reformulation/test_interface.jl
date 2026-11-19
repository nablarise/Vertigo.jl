# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Vertigo.Reformulation: compute_dual_bound_pure_master_contribution

# The contribution of `y` is its reduced cost times the bound that reduced cost
# points to: the lower bound when it is positive, the upper one when negative.
function test_reformulation_pure_master_contribution()
    # Every `assign` dual at 1 + 1e-12 gives `y` the reduced cost -3e-12, the
    # noise a solver leaves on a basic variable.
    noise = 1 + 1e-12
    # (name, bounds of `y`, contribution for the duals 0, 1, 2 and `noise`),
    # that is for the reduced costs 3, 0, -3 and about -3e-12.
    cases = [
        ("lower bound only", y -> set_lower_bound(y, 2), (6.0, 0.0, -Inf, 0.0)),
        ("non-negative", y -> set_lower_bound(y, 0), (0.0, 0.0, -Inf, 0.0)),
        (
            "both bounds",
            y -> (set_lower_bound(y, 2); set_upper_bound(y, 7)),
            (6.0, 0.0, -21.0, 0.0),
        ),
    ]
    for (name, set_bounds!, expected) in cases
        @testset "[pure_master_contribution] $name" begin
            decomp = build_pure_master_y_decomp(set_bounds!)
            for (dual, value) in zip((0.0, 1.0, 2.0, noise), expected)
                contribution = compute_dual_bound_pure_master_contribution(
                    decomp, _ -> dual
                )
                @test contribution ≈ value atol=1e-10
            end
        end
    end
end
