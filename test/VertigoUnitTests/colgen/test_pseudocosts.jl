# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.BranchCutPrice: PseudocostRecord, PseudocostTracker,
    update_pseudocosts!, estimate_score, is_reliable,
    global_average_pseudocost,
    BranchingCandidate, SBProbeResult, SBCandidateResult

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
        # Δ⁻ = 12.0 - 10.0 = 2.0, unit_down = 2.0 / 0.3
        @test rec.count_down == 1
        @test rec.sum_down ≈ 2.0 / 0.3
        # Δ⁺ = 14.0 - 10.0 = 4.0, unit_up = 4.0 / 0.7
        @test rec.count_up == 1
        @test rec.sum_up ≈ 4.0 / 0.7
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
        expected = (1 - mu) * 2.0 + mu * 4.0
        @test estimate_score(tracker, c) ≈ expected
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
        @test avg_down == 1.0
        @test avg_up == 1.0

        # Add one observation
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        update_pseudocosts!(tracker, c, result)

        avg_down, avg_up = global_average_pseudocost(tracker)
        @test avg_down ≈ 2.0 / 0.3
        @test avg_up ≈ 4.0 / 0.7
    end
end
