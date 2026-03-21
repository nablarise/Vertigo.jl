# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.BranchCutPrice: SBProbeResult, SBCandidateResult,
    sb_score, BranchingCandidate, find_fractional_variables,
    bp_master_model, bp_pool, bp_decomp, bp_branching_constraints,
    build_branching_terms, add_branching_constraint!,
    remove_branching_constraint!, BPSpace
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
end
