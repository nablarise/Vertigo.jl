# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Vertigo.Branching: solve_master_lp_only!, bp_master_model

function test_lp_probe_feasible()
    @testset "[solve_master_lp_only!] returns LP objective" begin
        inst = random_gap_instance(2, 5; seed=10)
        ws = build_gap_context(inst)
        run_column_generation(ws)
        backend = bp_master_model(ws)
        obj, is_inf = solve_master_lp_only!(backend)
        @test !is_inf
        @test !isnothing(obj)
        @test obj isa Float64
    end
end

function test_lp_probe_infeasible()
    @testset "[solve_master_lp_only!] detects infeasible LP" begin
        inst = random_gap_instance(2, 5; seed=10)
        ws = build_gap_context(inst)
        run_column_generation(ws)
        backend = bp_master_model(ws)

        # Add contradictory constraints to make the LP infeasible
        x = MOI.get(backend, MOI.ListOfVariableIndices())
        f = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, xi) for xi in x],
            0.0
        )
        MOI.add_constraint(backend, f, MOI.GreaterThan(1e8))
        MOI.add_constraint(backend, f, MOI.LessThan(-1e8))

        obj, is_inf = solve_master_lp_only!(backend)
        @test is_inf
        @test isnothing(obj)
    end
end

function test_lp_probe()
    test_lp_probe_feasible()
    test_lp_probe_infeasible()
end
