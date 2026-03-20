# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.ColGen: ColGenContext, max_cg_iterations,
    set_max_cg_iterations!

function test_max_cg_iterations()
    @testset "[max_cg_iterations] default value" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst)
        @test max_cg_iterations(ctx) == 1000
    end

    @testset "[max_cg_iterations] set and get" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst)
        set_max_cg_iterations!(ctx, 10)
        @test max_cg_iterations(ctx) == 10
    end

    @testset "[max_cg_iterations] constructor kwarg" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst; max_cg_iterations=50)
        @test max_cg_iterations(ctx) == 50
    end
end
