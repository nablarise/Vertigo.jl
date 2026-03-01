# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_ts_logger_default_protocol_methods()
    @testset "[ts_logger] default protocol methods" begin
        space = MockSearchSpace(max_depth=2)
        @test TreeSearch.ts_incumbent_value(space) === nothing
        @test TreeSearch.ts_best_dual_bound(space) === nothing
        @test TreeSearch.ts_is_minimization(space) == true
        @test TreeSearch.ts_nodes_explored(space) == 0
        @test TreeSearch.ts_search_status_message(space) ==
            "Search complete."
    end
end

function test_ts_logger_search_with_logger_context()
    @testset "[ts_logger] search with logger context" begin
        space = MockSearchSpace(max_depth=2)
        evaluator = MockEvaluator()
        ctx = TreeSearchLoggerContext(space, evaluator)

        result = redirect_stdout(devnull) do
            search(DepthFirstStrategy(), ctx)
        end

        # Same DFS traversal as without logger
        @test result.visited_values == [1, 2, 4, 5, 3, 6, 7]
    end
end

function test_ts_logger_feasible_with_logger()
    @testset "[ts_logger] feasible detection with logger" begin
        space = MockSearchSpace(
            max_depth=2, feasible_values=Set([2, 6])
        )
        evaluator = MockEvaluator()
        ctx = TreeSearchLoggerContext(space, evaluator)

        result = redirect_stdout(devnull) do
            search(DepthFirstStrategy(), ctx)
        end

        @test result.visited_values == [1, 2, 3, 6, 7]
        @test result.best == 2
    end
end

function test_tree_search_logger()
    test_ts_logger_default_protocol_methods()
    test_ts_logger_search_with_logger_context()
    test_ts_logger_feasible_with_logger()
end
