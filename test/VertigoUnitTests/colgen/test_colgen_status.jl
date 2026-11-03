# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Vertigo.ColGen: ColGenPhaseOutput, ColGenOutput, new_output, optimal, iteration_limit

function test_colgen_status()
    @testset "[colgen status] converging without a closed gap warns" begin
        inst = random_gap_instance(2, 5; seed=42)
        ws = build_gap_context(inst)
        phase_output(db; converged=true) =
            ColGenPhaseOutput(63.0, db, 10, false, converged, false, false)
        status(db; kw...) = new_output(ColGenOutput, ws, phase_output(db; kw...)).status

        @test (@test_logs status(63.0)) == optimal
        @test (@test_logs status(63.0 - 1e-7)) == optimal   # within tolerance
        @test (@test_logs status(60.0; converged=false)) == iteration_limit
        # Not proven, but the run goes on with the same status.
        @test (@test_logs (
            :warn, r"reported optimal, but the dual bound and the primal bound"
        ) status(-Inf)) == optimal
        @test (@test_logs (
            :warn, r"reported optimal, but the dual bound and the primal bound"
        ) status(60.0)) == optimal
    end
end
