# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.Branching: select_initial_candidates,
    score_candidate, filter_candidates, stop_phase,
    BranchingCandidate, SBProbeResult, SBCandidateResult,
    LPProbePhase, CGProbePhase,
    PseudocostTracker, estimate_score

function test_kernel()
    @testset "[select_initial_candidates] sorts by pseudocost" begin
        tracker = PseudocostTracker{Int}()
        c1 = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        c2 = BranchingCandidate(2, 3.5, 3.0, 4.0, 0.5)
        c3 = BranchingCandidate(3, 4.7, 4.0, 5.0, 0.3)
        result = select_initial_candidates(
            tracker, [c1, c2, c3], 2
        )
        @test length(result) == 2
    end

    @testset "[filter_candidates] keeps fraction" begin
        phase = LPProbePhase(keep_fraction=0.5)
        next = CGProbePhase()
        c1 = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        c2 = BranchingCandidate(2, 3.5, 3.0, 4.0, 0.5)
        c3 = BranchingCandidate(3, 4.7, 4.0, 5.0, 0.3)
        c4 = BranchingCandidate(4, 5.2, 5.0, 6.0, 0.2)
        scored = [
            (c1, 10.0), (c2, 8.0),
            (c3, 6.0), (c4, 4.0)
        ]
        result = filter_candidates(phase, next, scored)
        @test length(result) == 2
        @test result[1] === c1
        @test result[2] === c2
    end

    @testset "[filter_candidates] last phase keeps all" begin
        phase = CGProbePhase(keep_fraction=1.0)
        scored = [
            (BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3), 10.0),
            (BranchingCandidate(2, 3.5, 3.0, 4.0, 0.5), 8.0),
        ]
        result = filter_candidates(phase, nothing, scored)
        @test length(result) == 2
    end

    @testset "[stop_phase] lookahead triggers" begin
        phase = CGProbePhase(lookahead=2)
        @test !stop_phase(phase, 1, 10.0, 0)
        @test !stop_phase(phase, 2, 10.0, 1)
        @test stop_phase(phase, 3, 10.0, 2)
    end

    @testset "[stop_phase] LP phase no lookahead" begin
        phase = LPProbePhase(lookahead=0)
        @test !stop_phase(phase, 1, 10.0, 100)
    end

    @testset "[score_candidate] default product score" begin
        phase = CGProbePhase()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        mu = 1.0 / 6.0
        @test score_candidate(phase, result; mu=mu) ≈ 14.0 / 6.0 atol=1e-10
    end
end
