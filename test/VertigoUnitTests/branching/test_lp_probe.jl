# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Vertigo.Branching: solve_master_lp_only!, bp_master_model,
    probe_candidate, LPProbePhase, DefaultBranchingContext,
    bp_branching_constraints

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

function test_lp_probe_candidate_both_directions()
    @testset "[probe_candidate] LP probe leaves the master solvable" begin
        inst = random_gap_instance(2, 5; seed=10)
        space = BranchCutPriceWorkspace(
            build_gap_decomp(inst), BranchCutPriceConfig(node_limit=1)
        )
        run_column_generation(space.ws)
        backend = bp_master_model(space.ws)
        parent_lp, _ = solve_master_lp_only!(backend)
        primal = get_primal_solution(backend)
        candidate = first(find_fractional_variables(space.ws, primal; tol=1e-6))

        result = probe_candidate(
            DefaultBranchingContext(), LPProbePhase(), space, candidate, parent_lp
        )

        # Each probe tightens a minimization LP, so neither can lower its value.
        @test !result.left.is_infeasible
        @test result.left.dual_bound >= parent_lp - 1e-6
        @test !result.right.is_infeasible
        @test result.right.dual_bound >= parent_lp - 1e-6
        @test isempty(bp_branching_constraints(space.ws))

        # The probes removed their constraints, so the parent LP comes back.
        MOI.optimize!(backend)
        @test MOI.get(backend, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(backend, MOI.ObjectiveValue()) ≈ parent_lp atol=1e-6
    end
end

function test_lp_probe()
    test_lp_probe_feasible()
    test_lp_probe_infeasible()
    test_lp_probe_candidate_both_directions()
end
