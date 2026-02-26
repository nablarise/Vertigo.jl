# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module ColumnGenerationUnitTests

using Test
using ColumnGeneration

# ---------------------------------------------------------------------------
# ColumnGenerationProblem
# ---------------------------------------------------------------------------

function test_problem_constructor_ok()
    @testset "[problem] valid construction" begin
        obj = [1.0, 2.0]
        A = [1.0 0.0; 0.0 1.0]
        rhs = [3.0, 4.0]
        prob = ColumnGenerationProblem(obj, A, rhs, 100)

        @test prob.obj == obj
        @test prob.constraints == A
        @test prob.rhs == rhs
        @test prob.max_iterations == 100
    end
end

function test_solve_returns_nothing()
    @testset "[solve] placeholder returns nothing" begin
        prob = ColumnGenerationProblem([1.0], reshape([1.0], 1, 1), [1.0], 10)
        result = solve(prob)
        @test result === nothing
    end
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function run()
    test_problem_constructor_ok()
    test_solve_returns_nothing()
end

end # module ColumnGenerationUnitTests
