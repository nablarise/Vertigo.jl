# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ── Artificial variables e2e tests on GAP class A ────────────────────────

function _count_artificial_vars_in_model(model)
    return count(MOI.get(model, MOI.ListOfVariableIndices())) do v
        return startswith(MOI.get(model, MOI.VariableName(), v), "s_")
    end
end

function _count_tracked_artificial_vars(ws)
    return 2 * length(ws.eq_art_vars) + length(ws.leq_art_vars) + length(ws.geq_art_vars)
end

function test_artificial_vars_e2e()
    inst = parse_gap_file(get_gap_instance_path('A', 20, 100))

    @testset "[artificial vars] successive column generation runs do not stack them" begin
        bcp_ws = BranchCutPriceWorkspace(build_gap_decomp(inst), BranchCutPriceConfig())
        ws = bcp_ws.ws
        for _ in 1:3
            Vertigo.ColGen.run_column_generation(ws)
            @test _count_artificial_vars_in_model(
                Vertigo.BranchCutPrice.bp_master_model(ws)
            ) == _count_tracked_artificial_vars(ws)
        end
    end

    @testset "[artificial vars] restricted master incumbent is made of columns only" begin
        bcp_ws = BranchCutPriceWorkspace(
            build_gap_decomp(inst), BranchCutPriceConfig(node_limit=5)
        )
        output = run_branch_and_price(bcp_ws)
        if !isnothing(output.incumbent)
            @test output.incumbent.obj_value < 10000.0
            @test output.incumbent.obj_value >= output.best_dual_bound - 1e-6
        end
    end
end
