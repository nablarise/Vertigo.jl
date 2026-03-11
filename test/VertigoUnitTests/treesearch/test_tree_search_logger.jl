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
        @test TreeSearch.ts_open_node_count(space) == 0
        @test TreeSearch.ts_total_columns(space) == 0
        @test TreeSearch.ts_active_columns(space) == 0
        @test TreeSearch.ts_total_cuts(space) == 0
        node = TreeSearch.root_node(
            TreeSearch.NodeIdCounter(), (), (), nothing
        )
        @test TreeSearch.ts_branching_description(space, node) ===
            nothing
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

function test_ts_logger_constructor_backward_compat()
    @testset "[ts_logger] 2-arg constructor backward compat" begin
        space = MockSearchSpace(max_depth=1)
        evaluator = MockEvaluator()
        ctx = TreeSearchLoggerContext(space, evaluator)
        @test ctx.log_level == 1
    end
end

function test_ts_logger_constructor_3arg()
    @testset "[ts_logger] 3-arg constructor with log_level" begin
        space = MockSearchSpace(max_depth=1)
        evaluator = MockEvaluator()
        ctx = TreeSearchLoggerContext(space, evaluator, 2)
        @test ctx.log_level == 2
    end
end

function test_ts_logger_level2_traversal_order()
    @testset "[ts_logger] level 2 same traversal order" begin
        space = MockSearchSpace(max_depth=2)
        evaluator = MockEvaluator()
        ctx = TreeSearchLoggerContext(space, evaluator, 2)

        result = redirect_stdout(devnull) do
            search(DepthFirstStrategy(), ctx)
        end

        @test result.visited_values == [1, 2, 4, 5, 3, 6, 7]
    end
end

function test_ts_logger_level2_output_contains_banner()
    @testset "[ts_logger] level 2 output contains banner" begin
        space = MockSearchSpace(max_depth=2)
        evaluator = MockEvaluator()
        ctx = TreeSearchLoggerContext(space, evaluator, 2)

        pipe = Pipe()
        redirect_stdout(pipe) do
            search(DepthFirstStrategy(), ctx)
        end
        close(pipe.in)
        output = read(pipe, String)

        @test occursin("****", output)
        @test occursin("BaB tree root node", output)
        @test occursin("Node 1 done", output)
    end
end

function test_tree_search_logger()
    test_ts_logger_default_protocol_methods()
    test_ts_logger_search_with_logger_context()
    test_ts_logger_feasible_with_logger()
    test_ts_logger_constructor_backward_compat()
    test_ts_logger_constructor_3arg()
    test_ts_logger_level2_traversal_order()
    test_ts_logger_level2_output_contains_banner()
end
