# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Vertigo.Branching: BranchingLoggerContext, BranchingCandidate,
    SBProbeResult, SBCandidateResult, LPProbePhase, CGProbePhase,
    before_branching_selection, after_reliability_skip,
    after_candidate_probed,
    on_both_infeasible, after_phase_filter,
    after_branching_selection

function test_logger_before_branching_selection_prints_header()
    @testset "[branching_logger] before_branching_selection" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        before_branching_selection(lctx, BranchingCandidate[], [LPProbePhase()])
        @test String(take!(buf)) == "**** Strong branching ****\n"
        @test lctx.t0 > 0.0
    end
end

function test_logger_after_reliability_skip()
    @testset "[branching_logger] after_reliability_skip" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        lctx.t0 = time()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        after_reliability_skip(lctx, CGProbePhase(max_cg_iterations=5), 1, c, 4.5)
        output = String(take!(buf))
        @test occursin(
            r"^  \[CG\] cand\.  1 branch on 1 \(lhs=2\.3000\): reliable, score = 4\.50  <et=\d+\.\d+>$"m,
            output
        )
    end
end

function test_logger_after_candidate_probed()
    @testset "[branching_logger] after_candidate_probed" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        lctx.t0 = time()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        after_candidate_probed(lctx, LPProbePhase(), 1, c, 3.2, result)
        output = String(take!(buf))
        @test occursin(
            r"^  \[LP\] cand\.  1 branch on 1 \(lhs=2\.3000\): \[12\.0000, 14\.0000\], score = 3\.20  <et=\d+\.\d+>$"m,
            output
        )
    end
end

function test_logger_after_candidate_probed_infeasible_child()
    @testset "[branching_logger] after_candidate_probed infeasible child" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        lctx.t0 = time()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(nothing, nothing, true)
        result = SBCandidateResult(c, 10.0, left, right)
        after_candidate_probed(lctx, CGProbePhase(), 3, c, Inf, result)
        output = String(take!(buf))
        @test occursin(
            r"^  \[CG\] cand\.  3 branch on 1 \(lhs=2\.3000\): \[12\.0000, infeasible\], score = Inf  <et=\d+\.\d+>$"m,
            output
        )
    end
end

function test_logger_on_both_infeasible()
    @testset "[branching_logger] on_both_infeasible" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        on_both_infeasible(lctx, LPProbePhase(), 1, c)
        @test String(take!(buf)) == "  [LP] cand.  1 branch on 1: both infeasible\n"
    end
end

function test_logger_after_phase_filter()
    @testset "[branching_logger] after_phase_filter" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        after_phase_filter(lctx, "LP", 10, 3)
        @test String(take!(buf)) == "  [LP] filtered: 10 -> 3 candidates\n"
    end
end

function test_logger_after_branching_selection()
    @testset "[branching_logger] after_branching_selection" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        after_branching_selection(lctx, c, 7.5)
        @test String(take!(buf)) == "  SB selected: 1 (score = 7.50)\n"
    end
end

function test_branching_logger()
    test_logger_before_branching_selection_prints_header()
    test_logger_after_reliability_skip()
    test_logger_after_candidate_probed()
    test_logger_after_candidate_probed_infeasible_child()
    test_logger_on_both_infeasible()
    test_logger_after_phase_filter()
    test_logger_after_branching_selection()
end
