# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Vertigo.ColGen: compute_dual_bound, MasterDualSolution, DualMoiSolution
using Vertigo.Reformulation: TaggedCI, VI_GEQ, pure_master_coupling_coefficients

# `y ≥ 2` has no upper bound, so the master has no `y ≤ ub` constraint and
# no dual for it. With every `assign` dual at 0.5, the reduced cost of `y`
# is 1.5, which is also the dual of `y ≥ 2`. The Lagrangian bound is
# 3 × 0.5 + 1.5 × 2 = 4.5: the subproblems add nothing, their bounds and
# convexity duals being zero.
function test_colgen_dual_bound_pure_master_without_upper_bound()
    @testset "[dual_bounds] pure master variable without an upper bound" begin
        decomp = build_pure_master_y_decomp(y -> set_lower_bound(y, 2))
        y_var = only(pure_master_variables(decomp))
        duals = Dict{TaggedCI,Float64}(
            entry.constraint_id => 0.5
            for entry in pure_master_coupling_coefficients(decomp, y_var)
        )
        duals[TaggedCI(y_var.id.value, VI_GEQ)] = 1.5
        # Dual objective: 0.5 × 1 for each of the three `assign`, plus 1.5 × 2.
        dual_sol = MasterDualSolution(DualMoiSolution(4.5, duals), TaggedCI[])
        ws = ColGenWorkspace(decomp, ColGenConfig())
        sps_db = Dict(sp_id => 0.0 for sp_id in subproblem_ids(decomp))

        dual_bound = compute_dual_bound(ws, Phase2(), sps_db, dual_sol)

        @test dual_bound ≈ 4.5 atol=1e-10
    end
end
