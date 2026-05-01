# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Vertigo.Branching: SBProbeResult, SBCandidateResult,
    sb_score, BranchingCandidate, find_fractional_variables,
    bp_master_model, bp_pool, bp_decomp, bp_branching_constraints,
    build_branching_terms, add_branching_constraint!,
    remove_branching_constraint!,
    bp_ip_incumbent, bp_ip_primal_bound, run_sb_probe,
    MultiPhaseStrongBranching, CGProbePhase, select_branching_variable,
    BranchingResult, branching_ok,
    DefaultBranchingContext
using Vertigo.ColGen: max_cg_iterations
using Vertigo.Reformulation: get_primal_solution

function test_sb_score_both_feasible()
    @testset "[sb_score] both children feasible" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        # Δ⁻ = 12.0 - 10.0 = 2.0, Δ⁺ = 14.0 - 10.0 = 4.0
        # score = (1 - 1/6) * min(2,4) + (1/6) * max(2,4)
        #       = (5/6) * 2 + (1/6) * 4 = 10/6 + 4/6 = 14/6
        mu = 1.0 / 6.0
        expected = (1 - mu) * 2.0 + mu * 4.0
        @test sb_score(result) ≈ expected
    end
end

function test_sb_score_one_infeasible()
    @testset "[sb_score] one child infeasible" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(nothing, nothing, true)
        result = SBCandidateResult(c, 10.0, left, right)
        # Δ⁻ = 2.0, Δ⁺ = Inf
        # score = (5/6) * 2.0 + (1/6) * Inf = Inf
        @test sb_score(result) == Inf
    end
end

function test_sb_score_both_infeasible()
    @testset "[sb_score] both children infeasible" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(nothing, nothing, true)
        right = SBProbeResult(nothing, nothing, true)
        result = SBCandidateResult(c, 10.0, left, right)
        @test sb_score(result) == Inf
    end
end

function test_sb_score_custom_mu()
    @testset "[sb_score] custom mu" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        mu = 0.25
        expected = (1 - mu) * 2.0 + mu * 4.0
        @test sb_score(result; mu=mu) ≈ expected
    end
end

function test_branching_constraint_add_remove()
    @testset "[branching_constraint] add and remove" begin
        inst = random_gap_instance(2, 5; seed=10)
        ws = build_gap_context(inst)
        run_column_generation(ws)

        backend = bp_master_model(ws)
        bcs = bp_branching_constraints(ws)
        @test isempty(bcs)

        pool = bp_pool(ws)
        decomp = bp_decomp(ws)
        primal = get_primal_solution(backend)
        candidates = find_fractional_variables(
            ws, primal; tol=1e-6
        )
        @test !isempty(candidates)
        candidate = first(candidates)

        terms = build_branching_terms(
            decomp, pool, candidate.orig_var
        )
        ci = add_branching_constraint!(
            backend, ws, terms,
            MOI.LessThan(candidate.floor_val),
            candidate.orig_var
        )
        @test length(bcs) == 1
        @test MOI.is_valid(backend, ci)

        remove_branching_constraint!(backend, ws, ci)
        @test isempty(bcs)
        @test !MOI.is_valid(backend, ci)
    end
end

function test_run_sb_probe_returns_dual_bounds()
    @testset "[run_sb_probe] returns dual bounds" begin
        inst = random_gap_instance(2, 5; seed=10)
        decomp = build_gap_decomp(inst)
        space = BranchCutPriceWorkspace(
            decomp, BranchCutPriceConfig(node_limit=1)
        )
        ws = space.ws
        cg_out = run_column_generation(ws)
        parent_lp = cg_out.master_lp_obj

        backend = bp_master_model(ws)
        primal = get_primal_solution(backend)
        candidates = find_fractional_variables(
            ws, primal; tol=1e-6
        )
        candidate = first(candidates)

        result = run_sb_probe(DefaultBranchingContext(), CGProbePhase(max_cg_iterations=10), space, candidate, 10, parent_lp)
        @test result isa SBCandidateResult
        @test result.parent_lp_obj ≈ parent_lp
        # Each direction should produce a dual bound or be infeasible
        @test !isnothing(result.left.dual_bound) ||
              result.left.is_infeasible
        @test !isnothing(result.right.dual_bound) ||
              result.right.is_infeasible
    end
end

function test_run_sb_probe_restores_state()
    @testset "[run_sb_probe] restores context state" begin
        inst = random_gap_instance(2, 5; seed=10)
        decomp = build_gap_decomp(inst)
        space = BranchCutPriceWorkspace(
            decomp, BranchCutPriceConfig(node_limit=1)
        )
        ws = space.ws
        cg_out = run_column_generation(ws)
        parent_lp = cg_out.master_lp_obj

        backend = bp_master_model(ws)
        primal = get_primal_solution(backend)
        candidates = find_fractional_variables(
            ws, primal; tol=1e-6
        )
        candidate = first(candidates)

        # Save state before probe
        orig_max_iter = max_cg_iterations(ws)
        orig_ip_inc = bp_ip_incumbent(ws)
        orig_ip_bound = bp_ip_primal_bound(ws)
        orig_n_bcs = length(bp_branching_constraints(ws))

        run_sb_probe(DefaultBranchingContext(), CGProbePhase(max_cg_iterations=10), space, candidate, 10, parent_lp)

        # State must be restored
        @test max_cg_iterations(ws) == orig_max_iter
        @test bp_ip_incumbent(ws) === orig_ip_inc
        @test bp_ip_primal_bound(ws) === orig_ip_bound
        @test length(bp_branching_constraints(ws)) == orig_n_bcs
    end
end

function test_multi_phase_sb_e2e()
    @testset "[MultiPhaseStrongBranching] e2e small GAP finds optimal" begin
        inst = random_gap_instance(2, 4; seed=10)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp,
            BranchCutPriceConfig(
                node_limit=100,
                branching_strategy=MultiPhaseStrongBranching(
                    max_candidates=3,
                    phases=[CGProbePhase(
                        max_cg_iterations=5, lookahead=0
                    )]
                )
            )
        )
        output = run_branch_and_price(bcp_ws)
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
    end
end

function test_multi_phase_sb_selects_variable()
    @testset "[MultiPhaseStrongBranching] selects branching variable" begin
        inst = random_gap_instance(2, 5; seed=10)
        decomp = build_gap_decomp(inst)
        sb = MultiPhaseStrongBranching(
            max_candidates=5,
            phases=[CGProbePhase(
                max_cg_iterations=10, lookahead=0
            )]
        )
        space = BranchCutPriceWorkspace(
            decomp,
            BranchCutPriceConfig(node_limit=1, branching_strategy=sb)
        )
        run_column_generation(space.ws)
        primal = get_primal_solution(bp_master_model(space.ws))

        result = select_branching_variable(
            sb, space, nothing, primal
        )
        @test result.status == branching_ok
        # Should pick a fractional variable
        frac = result.value - floor(result.value)
        @test frac > 1e-6
        @test frac < 1.0 - 1e-6
    end
end

# ────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────

function test_strong_branching()
    test_sb_score_both_feasible()
    test_sb_score_one_infeasible()
    test_sb_score_both_infeasible()
    test_sb_score_custom_mu()
    test_branching_constraint_add_remove()
    test_run_sb_probe_returns_dual_bounds()
    test_run_sb_probe_restores_state()
    test_multi_phase_sb_e2e()
    test_multi_phase_sb_selects_variable()
end
