# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.Branching: PseudocostRecord, PseudocostTracker,
    update_pseudocosts!, estimate_score, is_reliable,
    global_average_pseudocost,
    BranchingCandidate, SBProbeResult, SBCandidateResult,
    MultiPhaseStrongBranching, CGProbePhase, select_branching_variable,
    bp_master_model, branching_ok
using Vertigo.BranchCutPrice: BPSpace, BPNodeData
using Vertigo.Reformulation: get_primal_solution

function test_pseudocosts()
    @testset "[pseudocosts] cold start" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        @test !is_reliable(tracker, c)
        @test !haskey(tracker.records, 1)
    end

    @testset "[pseudocosts] update from probe result" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)

        update_pseudocosts!(tracker, c, result)

        rec = tracker.records[1]
        # frac_part = 0.3
        # Δ⁻ = 12.0 - 10.0 = 2.0, unit_down = 2.0 / 0.3 ≈ 6.667
        @test rec.count_down == 1
        @test rec.sum_down ≈ 6.666666666666667 atol=1e-10
        # Δ⁺ = 14.0 - 10.0 = 4.0, unit_up = 4.0 / 0.7 ≈ 5.714
        @test rec.count_up == 1
        @test rec.sum_up ≈ 5.714285714285714 atol=1e-10
    end

    @testset "[pseudocosts] skip infeasible side" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(nothing, nothing, true)
        result = SBCandidateResult(c, 10.0, left, right)

        update_pseudocosts!(tracker, c, result)

        rec = tracker.records[1]
        @test rec.count_down == 1
        @test rec.count_up == 0
        @test rec.sum_up == 0.0
    end

    @testset "[pseudocosts] multiple observations" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)

        for i in 1:3
            left = SBProbeResult(
                10.0 + Float64(i), 10.0 + Float64(i), false
            )
            right = SBProbeResult(
                10.0 + 2.0 * Float64(i),
                10.0 + 2.0 * Float64(i), false
            )
            result = SBCandidateResult(c, 10.0, left, right)
            update_pseudocosts!(tracker, c, result)
        end

        rec = tracker.records[1]
        @test rec.count_down == 3
        @test rec.count_up == 3
    end

    @testset "[pseudocosts] estimate_score with observations" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        update_pseudocosts!(tracker, c, result)

        # mean_unit_down = (2.0/0.3) / 1 = 6.667
        # mean_unit_up = (4.0/0.7) / 1 = 5.714
        # score_down = 6.667 * 0.3 = 2.0
        # score_up = 5.714 * 0.7 = 4.0
        mu = 1.0 / 6.0
        # (1 - mu) * min(2.0, 4.0) + mu * max(2.0, 4.0) = 14/6
        @test estimate_score(tracker, c) ≈ 14.0 / 6.0 atol=1e-10
    end

    @testset "[pseudocosts] estimate_score uses global average fallback" begin
        tracker = PseudocostTracker{Int}()
        # Var 1 has observations
        c1 = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c1, 10.0, left, right)
        update_pseudocosts!(tracker, c1, result)

        # Var 2 has no observations — should use global average
        c2 = BranchingCandidate(2, 3.4, 3.0, 4.0, 0.4)
        score = estimate_score(tracker, c2)
        @test score > 0.0  # not zero — uses global average
    end

    @testset "[pseudocosts] is_reliable" begin
        tracker = PseudocostTracker{Int}(
            reliability_threshold=2
        )
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        @test !is_reliable(tracker, c)

        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)

        update_pseudocosts!(tracker, c, result)
        @test !is_reliable(tracker, c)

        update_pseudocosts!(tracker, c, result)
        @test is_reliable(tracker, c)
    end

    @testset "[pseudocosts] global_average_pseudocost" begin
        tracker = PseudocostTracker{Int}()
        # Empty tracker — returns (1.0, 1.0) per Achterberg §2.2
        avg_down, avg_up = global_average_pseudocost(tracker)
        @test avg_down ≈ 1.0 atol=1e-10
        @test avg_up ≈ 1.0 atol=1e-10

        # Add one observation
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        update_pseudocosts!(tracker, c, result)

        avg_down, avg_up = global_average_pseudocost(tracker)
        # avg_down = sum_down / count = (2.0/0.3) / 1 ≈ 6.667
        @test avg_down ≈ 6.666666666666667 atol=1e-10
        # avg_up = sum_up / count = (4.0/0.7) / 1 ≈ 5.714
        @test avg_up ≈ 5.714285714285714 atol=1e-10
    end

    @testset "[MultiPhaseStrongBranching] selects variable with cg_output" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        cg_out = run_column_generation(ctx)

        primal = get_primal_solution(bp_master_model(ctx))
        rb = MultiPhaseStrongBranching(
            max_candidates=10,
            phases=[CGProbePhase(
                max_cg_iterations=5, lookahead=8
            )],
            reliability_threshold=2
        )
        space = BPSpace(
            ctx; node_limit=1, branching_strategy=rb
        )

        # Create a mock node with cg_output
        node_data = BPNodeData()
        node_data.cg_output = cg_out
        mock_node = (user_data=node_data,)

        result = select_branching_variable(
            space.branching_strategy, space, mock_node, primal
        )
        @test result.status == branching_ok
        frac = result.value - floor(result.value)
        @test frac > 1e-6
        @test frac < 1.0 - 1e-6
    end

    @testset "[MultiPhaseStrongBranching] lookahead stops early" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        cg_out = run_column_generation(ctx)
        primal = get_primal_solution(bp_master_model(ctx))

        # lookahead=1, all unreliable -> at most 2 probed
        rb = MultiPhaseStrongBranching(
            max_candidates=100,
            phases=[CGProbePhase(
                max_cg_iterations=5, lookahead=1
            )],
            reliability_threshold=100
        )
        space = BPSpace(
            ctx; node_limit=1, branching_strategy=rb
        )
        strategy = space.branching_strategy
        node_data = BPNodeData()
        node_data.cg_output = cg_out
        mock_node = (user_data=node_data,)

        result = select_branching_variable(
            strategy, space, mock_node, primal
        )
        @test result.status == branching_ok

        # Verify lookahead actually cut the loop
        n_probed = length(strategy.pseudocosts.records)
        @test n_probed <= 2
    end

    @testset "[MultiPhaseStrongBranching] e2e small GAP" begin
        inst = random_gap_instance(2, 5; seed=10)
        rb = MultiPhaseStrongBranching(
            max_candidates=10,
            phases=[CGProbePhase(
                max_cg_iterations=5, lookahead=8
            )],
            reliability_threshold=2
        )
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx;
            node_limit=100,
            branching_strategy=rb
        )
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
        # Broken: run_branch_and_price reconstructs the strategy,
        # so rb.pseudocosts is never updated. See #37.
        @test_broken !isempty(rb.pseudocosts.records)
    end
end
