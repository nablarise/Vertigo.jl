# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.Branching: BranchingLoggerContext, BranchingCandidate,
    SBProbeResult, SBCandidateResult, LPProbePhase, CGProbePhase,
    before_branching_selection, after_candidate_eval,
    on_both_infeasible, after_phase_filter,
    after_branching_selection

function test_logger_before_branching_selection_prints_header()
    @testset "[branching_logger] before_branching_selection prints header" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        candidates = BranchingCandidate[]
        phases = [LPProbePhase()]
        before_branching_selection(lctx, candidates, phases)
        output = String(take!(buf))
        @test contains(output, "Strong branching")
        @test lctx.t0 > 0.0
    end
end

function test_logger_after_candidate_eval_reliable()
    @testset "[branching_logger] after_candidate_eval reliable skip" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        lctx.t0 = time()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        phase = CGProbePhase(max_cg_iterations=5)
        after_candidate_eval(lctx, phase, 1, c, 4.5, :reliable)
        output = String(take!(buf))
        @test contains(output, "reliable")
        @test contains(output, "4.50")
    end
end

function test_logger_after_candidate_eval_probed()
    @testset "[branching_logger] after_candidate_eval with probe result" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        lctx.t0 = time()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        phase = LPProbePhase()
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        after_candidate_eval(lctx, phase, 1, c, 3.2, result)
        output = String(take!(buf))
        @test contains(output, "12.0000")
        @test contains(output, "14.0000")
        @test contains(output, "3.20")
    end
end

function test_logger_on_both_infeasible()
    @testset "[branching_logger] on_both_infeasible" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        phase = LPProbePhase()
        on_both_infeasible(lctx, phase, 1, c)
        output = String(take!(buf))
        @test contains(output, "both infeasible")
    end
end

function test_logger_after_phase_filter()
    @testset "[branching_logger] after_phase_filter" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        after_phase_filter(lctx, "LP", 10, 3)
        output = String(take!(buf))
        @test contains(output, "LP")
        @test contains(output, "10 -> 3")
    end
end

function test_logger_after_branching_selection()
    @testset "[branching_logger] after_branching_selection" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        after_branching_selection(lctx, c, 7.5)
        output = String(take!(buf))
        @test contains(output, "SB selected")
        @test contains(output, "7.50")
    end
end

function test_branching_logger()
    test_logger_before_branching_selection_prints_header()
    test_logger_after_candidate_eval_reliable()
    test_logger_after_candidate_eval_probed()
    test_logger_on_both_infeasible()
    test_logger_after_phase_filter()
    test_logger_after_branching_selection()
end
