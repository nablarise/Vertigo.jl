# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

struct MockDiff
    delta::Int
end
const MOCK_EMPTY = MockDiff(0)

mutable struct MockUserData
    value::Int
end

"""
A mock search space that builds a binary tree of fixed depth.

At each node, two children are created:
- left child: value = parent.value * 2
- right child: value = parent.value * 2 + 1

Root value = 1, left child = 2, right child = 3, left-left = 4, etc.
"""
mutable struct MockSearchSpace <: TreeSearch.AbstractSearchSpace
    max_depth::Int
    counter::NodeIdCounter
    visited_ids::Vector{Int}
    visited_values::Vector{Int}
    transitions::Vector{Tuple{Int,Int}}
    current_model_value::Ref{Int}
    feasible_values::Set{Int}
    best_solution::Ref{Int}
end

function MockSearchSpace(; max_depth=3, feasible_values=Set{Int}())
    return MockSearchSpace(
        max_depth,
        NodeIdCounter(),
        Int[], Int[],
        Tuple{Int,Int}[],
        Ref(1),
        feasible_values,
        Ref(typemax(Int))
    )
end

function TreeSearch.new_root(space::MockSearchSpace)
    return root_node(space.counter, MOCK_EMPTY, MOCK_EMPTY, MockUserData(1))
end

TreeSearch.stop(::MockSearchSpace, _) = false

function TreeSearch.output(space::MockSearchSpace)
    return (visited_ids=space.visited_ids,
            visited_values=space.visited_values,
            transitions=space.transitions,
            best=space.best_solution[])
end

function TreeSearch.transition!(space::MockSearchSpace, current::SearchNode, next::SearchNode)
    push!(space.transitions, (current.id, next.id))
    transition_to!(current, next,
        diff -> (space.current_model_value[] += diff.delta),
        diff -> (space.current_model_value[] += diff.delta)
    )
end

function TreeSearch.branch!(space::MockSearchSpace, node::SearchNode)
    if node.depth >= space.max_depth
        return SearchNode{MockDiff, MockUserData}[]
    end

    parent_val = node.user_data.value
    left_val = parent_val * 2
    left_delta = left_val - parent_val
    left = child_node(space.counter, node,
        MockDiff(left_delta), MockDiff(-left_delta),
        user_data=MockUserData(left_val))

    right_val = parent_val * 2 + 1
    right_delta = right_val - parent_val
    right = child_node(space.counter, node,
        MockDiff(right_delta), MockDiff(-right_delta),
        user_data=MockUserData(right_val))

    return [left, right]
end

function TreeSearch.on_feasible_solution!(space::MockSearchSpace, node::SearchNode)
    val = node.user_data.value
    if val < space.best_solution[]
        space.best_solution[] = val
    end
end

struct MockEvaluator <: TreeSearch.AbstractNodeEvaluator end

function TreeSearch.evaluate!(::MockEvaluator, space::MockSearchSpace, node::SearchNode)
    push!(space.visited_ids, node.id)
    push!(space.visited_values, node.user_data.value)

    if node.user_data.value ∈ space.feasible_values
        return FEASIBLE
    end
    if node.depth >= space.max_depth
        return CUTOFF
    end
    return BRANCH
end

struct SelectiveCutoffEvaluator <: TreeSearch.AbstractNodeEvaluator
    cutoff_values::Set{Int}
end

function TreeSearch.evaluate!(eval::SelectiveCutoffEvaluator, space::MockSearchSpace, node::SearchNode)
    push!(space.visited_ids, node.id)
    push!(space.visited_values, node.user_data.value)

    if node.user_data.value ∈ eval.cutoff_values
        return CUTOFF
    end
    if node.depth >= space.max_depth
        return CUTOFF
    end
    return BRANCH
end

mutable struct BoundedSearchSpace <: TreeSearch.AbstractSearchSpace
    inner::MockSearchSpace
    max_evals::Int
    eval_count::Int
end

TreeSearch.new_root(s::BoundedSearchSpace) = TreeSearch.new_root(s.inner)
TreeSearch.stop(s::BoundedSearchSpace, _) = s.eval_count >= s.max_evals
TreeSearch.output(s::BoundedSearchSpace) = TreeSearch.output(s.inner)
TreeSearch.transition!(s::BoundedSearchSpace, c, n) = TreeSearch.transition!(s.inner, c, n)
TreeSearch.branch!(s::BoundedSearchSpace, node) = TreeSearch.branch!(s.inner, node)

struct BoundedEvaluator <: TreeSearch.AbstractNodeEvaluator end

function TreeSearch.evaluate!(::BoundedEvaluator, space::BoundedSearchSpace, node::SearchNode)
    space.eval_count += 1
    push!(space.inner.visited_ids, node.id)
    push!(space.inner.visited_values, node.user_data.value)
    if node.depth >= space.inner.max_depth
        return CUTOFF
    end
    return BRANCH
end

function test_treesearch_loop_dfs_order()
    @testset "[treesearch] DFS traversal order" begin
        # Binary tree depth 2:
        #       1
        #      / \
        #     2   3
        #    / \ / \
        #   4  5 6  7
        space = MockSearchSpace(max_depth=2)
        result = search(DepthFirstStrategy(), space, MockEvaluator())

        # DFS pre-order left-first: 1, 2, 4, 5, 3, 6, 7
        @test result.visited_values == [1, 2, 4, 5, 3, 6, 7]
    end
end

function test_treesearch_loop_bfs_order()
    @testset "[treesearch] BFS traversal order" begin
        space = MockSearchSpace(max_depth=2)
        result = search(BreadthFirstStrategy(), space, MockEvaluator())

        # BFS level-order: 1, 2, 3, 4, 5, 6, 7
        @test result.visited_values == [1, 2, 3, 4, 5, 6, 7]
    end
end

function test_treesearch_loop_cutoff_pruning()
    @testset "[treesearch] CUTOFF pruning" begin
        space = MockSearchSpace(max_depth=2)
        eval = SelectiveCutoffEvaluator(Set([2]))

        result = search(DepthFirstStrategy(), space, eval)

        # DFS: visit 1, 2 (CUTOFF), then 3, 6, 7
        @test result.visited_values == [1, 2, 3, 6, 7]
        @test 4 ∉ result.visited_values
        @test 5 ∉ result.visited_values
    end
end

function test_treesearch_loop_feasible_detection()
    @testset "[treesearch] FEASIBLE detection" begin
        space = MockSearchSpace(max_depth=2, feasible_values=Set([2, 6]))
        result = search(DepthFirstStrategy(), space, MockEvaluator())

        # Node 2 is feasible (no children), node 6 is feasible.
        # DFS: 1 (branch), 2 (feasible → no children), 3 (branch), 6 (feasible), 7 (cutoff)
        @test result.visited_values == [1, 2, 3, 6, 7]
        @test result.best == 2
        @test 4 ∉ result.visited_values
        @test 5 ∉ result.visited_values
    end
end

function test_treesearch_loop_stop_criterion()
    @testset "[treesearch] stop criterion" begin
        inner = MockSearchSpace(max_depth=3)
        bounded = BoundedSearchSpace(inner, 3, 0)

        result = search(DepthFirstStrategy(), bounded, BoundedEvaluator())

        @test length(result.visited_values) == 3
    end
end

function test_treesearch_loop_dfs_transitions()
    @testset "[treesearch] DFS transitions" begin
        space = MockSearchSpace(max_depth=1)
        result = search(DepthFirstStrategy(), space, MockEvaluator())

        @test result.visited_values == [1, 2, 3]
        @test length(result.transitions) == 2
        @test result.transitions[1][2] == space.visited_ids[2]
        @test result.transitions[2][2] == space.visited_ids[3]
    end
end

function test_treesearch_search_loop()
    test_treesearch_loop_dfs_order()
    test_treesearch_loop_bfs_order()
    test_treesearch_loop_cutoff_pruning()
    test_treesearch_loop_feasible_detection()
    test_treesearch_loop_stop_criterion()
    test_treesearch_loop_dfs_transitions()
end
