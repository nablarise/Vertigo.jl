# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

struct SDiff
    delta::Int
end
const SEMPTY = SDiff(0)

mutable struct SData
    value::Int
end

mutable struct StrategyTestSpace <: TreeSearch.AbstractSearchSpace
    max_depth::Int
    counter::NodeIdCounter
    visited_values::Vector{Int}
    model_val::Ref{Int}
end

function StrategyTestSpace(; max_depth=2)
    StrategyTestSpace(max_depth, NodeIdCounter(), Int[], Ref(1))
end

function TreeSearch.new_root(space::StrategyTestSpace)
    return root_node(space.counter, SEMPTY, SEMPTY, SData(1), dual_bound=0.0)
end

TreeSearch.stop(::StrategyTestSpace, _) = false

TreeSearch.output(space::StrategyTestSpace) = space.visited_values

function TreeSearch.transition!(space::StrategyTestSpace, current::SearchNode, next::SearchNode)
    transition_to!(current, next,
        diff -> (space.model_val[] += diff.delta),
        diff -> (space.model_val[] += diff.delta)
    )
end

function TreeSearch.branch!(space::StrategyTestSpace, node::SearchNode)
    if node.depth >= space.max_depth
        return SearchNode{SDiff, SData}[]
    end
    pv = node.user_data.value
    lv = pv * 2
    rv = pv * 2 + 1
    left = child_node(space.counter, node,
        SDiff(lv - pv), SDiff(pv - lv),
        dual_bound = Float64(lv),
        user_data = SData(lv))
    right = child_node(space.counter, node,
        SDiff(rv - pv), SDiff(pv - rv),
        dual_bound = Float64(rv),
        user_data = SData(rv))
    return [left, right]
end

struct StrategyEvaluator <: TreeSearch.AbstractNodeEvaluator end

function TreeSearch.evaluate!(::StrategyEvaluator, space::StrategyTestSpace, node::SearchNode)
    push!(space.visited_values, node.user_data.value)
    if node.depth >= space.max_depth
        return CUTOFF
    end
    return BRANCH
end

# Must be at module level — Julia does not allow struct definitions inside functions.
struct ValuePriorityStrategy <: AbstractBestFirstStrategy end
TreeSearch.get_priority(::ValuePriorityStrategy, node::SearchNode) = Float64(node.user_data.value)

function test_treesearch_strategies_best_first()
    @testset "[treesearch] best-first priority order" begin
        # Tree with dual bounds:
        #       1 (db=0)
        #      / \
        #     2   3     (db=2, db=3)
        #    / \ / \
        #   4  5 6  7   (db=4,5,6,7)
        #
        # Best-first: 1(db=0), 2(db=2), 3(db=3), 4(db=4), 5(db=5), 6(db=6), 7(db=7)
        space = StrategyTestSpace(max_depth=2)
        visited = search(DualBoundBestFirstStrategy(), space, StrategyEvaluator())

        @test visited[1] == 1
        @test visited[2] == 2
        @test visited[3] == 3
        @test Set(visited[4:7]) == Set([4, 5, 6, 7])
        @test visited[4] == 4
        @test visited[5] == 5
        @test visited[6] == 6
        @test visited[7] == 7
    end
end

function test_treesearch_strategies_beam_search()
    @testset "[treesearch] beam search width limit" begin
        space = StrategyTestSpace(max_depth=2)
        strategy = BeamSearchStrategy(ValuePriorityStrategy(), 2)
        visited = search(strategy, space, StrategyEvaluator())

        # Depth 0: evaluate root (value=1) → generates 2, 3
        # Depth 1: beam width = 2, evaluate 2 then 3 → generates 4,5 and 6,7
        # Depth 2: beam width = 2, evaluate 4 then 5 (best 2 of {4,5,6,7})
        # 6 and 7 are pruned by the beam
        @test visited[1] == 1
        @test 2 ∈ visited
        @test 3 ∈ visited
        @test 4 ∈ visited
        @test 5 ∈ visited
        @test 6 ∉ visited
        @test 7 ∉ visited
    end
end

function test_treesearch_strategies_bfs_consistency()
    @testset "[treesearch] BFS model state consistency" begin
        space = StrategyTestSpace(max_depth=2)
        visited = search(BreadthFirstStrategy(), space, StrategyEvaluator())

        @test visited == [1, 2, 3, 4, 5, 6, 7]
    end
end

function test_treesearch_strategies_dfs_child_order()
    @testset "[treesearch] DFS first child first" begin
        space = StrategyTestSpace(max_depth=2)
        visited = search(DepthFirstStrategy(), space, StrategyEvaluator())

        @test visited == [1, 2, 4, 5, 3, 6, 7]
    end
end

function test_treesearch_strategies()
    test_treesearch_strategies_best_first()
    test_treesearch_strategies_beam_search()
    test_treesearch_strategies_bfs_consistency()
    test_treesearch_strategies_dfs_child_order()
end
