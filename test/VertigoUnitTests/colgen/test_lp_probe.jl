# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.Branching: solve_master_lp_only!, bp_master_model

function test_lp_probe()
    @testset "[solve_master_lp_only!] returns LP objective" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)
        backend = bp_master_model(ctx)
        obj, is_inf = solve_master_lp_only!(backend)
        @test !is_inf
        @test !isnothing(obj)
        @test obj isa Float64
    end
end
