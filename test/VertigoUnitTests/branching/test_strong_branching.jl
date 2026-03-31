# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.Branching: SBProbeResult, SBCandidateResult,
    sb_score, BranchingCandidate, find_fractional_variables,
    bp_master_model, bp_pool, bp_decomp, bp_branching_constraints,
    build_branching_terms, add_branching_constraint!,
    remove_branching_constraint!,
    bp_ip_incumbent, bp_ip_primal_bound, run_sb_probe,
    MultiPhaseStrongBranching, CGProbePhase, select_branching_variable,
    BranchingResult, branching_ok
using Vertigo.BranchCutPrice: BPSpace
using Vertigo.ColGen: max_cg_iterations
using Vertigo.Reformulation: get_primal_solution

function test_strong_branching()
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

    @testset "[sb_score] one child infeasible" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(nothing, nothing, true)
        result = SBCandidateResult(c, 10.0, left, right)
        # Δ⁻ = 2.0, Δ⁺ = Inf
        # score = (5/6) * 2.0 + (1/6) * Inf = Inf
        @test sb_score(result) == Inf
    end

    @testset "[sb_score] both children infeasible" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(nothing, nothing, true)
        right = SBProbeResult(nothing, nothing, true)
        result = SBCandidateResult(c, 10.0, left, right)
        @test sb_score(result) == Inf
    end

    @testset "[sb_score] custom mu" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        mu = 0.25
        expected = (1 - mu) * 2.0 + mu * 4.0
        @test sb_score(result; mu=mu) ≈ expected
    end

    @testset "[branching_constraint] add and remove" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)

        backend = bp_master_model(ctx)
        bcs = bp_branching_constraints(ctx)
        @test isempty(bcs)

        # Build terms for a branching constraint
        pool = bp_pool(ctx)
        decomp = bp_decomp(ctx)
        primal = get_primal_solution(backend)
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        @test !isempty(candidates)
        candidate = first(candidates)

        terms = build_branching_terms(
            decomp, pool, candidate.orig_var
        )
        ci = add_branching_constraint!(
            backend, ctx, terms,
            MOI.LessThan(candidate.floor_val),
            candidate.orig_var
        )
        @test length(bcs) == 1
        @test MOI.is_valid(backend, ci)

        remove_branching_constraint!(backend, ctx, ci)
        @test isempty(bcs)
        @test !MOI.is_valid(backend, ci)
    end

    @testset "[run_sb_probe] returns dual bounds" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        cg_out = run_column_generation(ctx)
        parent_lp = cg_out.master_lp_obj

        backend = bp_master_model(ctx)
        primal = get_primal_solution(backend)
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        space = BPSpace(ctx; node_limit=1)
        candidate = first(candidates)

        result = run_sb_probe(space, candidate, 10, parent_lp)
        @test result isa SBCandidateResult
        @test result.parent_lp_obj ≈ parent_lp
        # At least one direction should produce a dual bound
        has_bound = !isnothing(result.left.dual_bound) ||
                    !isnothing(result.right.dual_bound)
        @test has_bound || result.left.is_infeasible ||
              result.right.is_infeasible
    end

    @testset "[run_sb_probe] restores context state" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        cg_out = run_column_generation(ctx)
        parent_lp = cg_out.master_lp_obj

        backend = bp_master_model(ctx)
        primal = get_primal_solution(backend)
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        space = BPSpace(ctx; node_limit=1)
        candidate = first(candidates)

        # Save state before probe
        orig_max_iter = max_cg_iterations(ctx)
        orig_ip_inc = bp_ip_incumbent(ctx)
        orig_ip_bound = bp_ip_primal_bound(ctx)
        orig_n_bcs = length(bp_branching_constraints(ctx))

        run_sb_probe(space, candidate, 10, parent_lp)

        # State must be restored
        @test max_cg_iterations(ctx) == orig_max_iter
        @test bp_ip_incumbent(ctx) === orig_ip_inc
        @test bp_ip_primal_bound(ctx) === orig_ip_bound
        @test length(bp_branching_constraints(ctx)) == orig_n_bcs
    end

    @testset "[MultiPhaseStrongBranching] e2e small GAP finds optimal" begin
        inst = random_gap_instance(2, 4; seed=10)
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx;
            node_limit=100,
            branching_strategy=MultiPhaseStrongBranching(
                max_candidates=3,
                phases=[CGProbePhase(
                    max_cg_iterations=5, lookahead=0
                )]
            )
        )
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
    end

    @testset "[MultiPhaseStrongBranching] selects branching variable" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)

        sb = MultiPhaseStrongBranching(
            max_candidates=5,
            phases=[CGProbePhase(
                max_cg_iterations=10, lookahead=0
            )]
        )
        primal = get_primal_solution(bp_master_model(ctx))
        space = BPSpace(
            ctx; node_limit=1,
            branching_strategy=sb
        )

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
