# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# Helper: simple diff type for testing.
# A diff is just a number that gets added to a "model" (an integer accumulator).
# Forward adds, backward subtracts.
struct TestDiff
    delta::Int
end

const EMPTY_DIFF = TestDiff(0)

"""
Build the test tree:

         R (root)
        / \\
       A   B
      / \\   \\
     C   D    E
          \\
           F

Each edge carries a diff: forward = +value, backward = -value.
"""
function build_test_tree()
    counter = NodeIdCounter()

    R = root_node(counter, EMPTY_DIFF, EMPTY_DIFF, nothing)
    A = child_node(counter, R, TestDiff(1), TestDiff(-1), user_data=nothing)
    B = child_node(counter, R, TestDiff(2), TestDiff(-2), user_data=nothing)
    C = child_node(counter, A, TestDiff(10), TestDiff(-10), user_data=nothing)
    D = child_node(counter, A, TestDiff(20), TestDiff(-20), user_data=nothing)
    E = child_node(counter, B, TestDiff(30), TestDiff(-30), user_data=nothing)
    F = child_node(counter, D, TestDiff(100), TestDiff(-100), user_data=nothing)

    return (; R, A, B, C, D, E, F, counter)
end

function test_treesearch_node_construction()
    @testset "[treesearch] tree construction" begin
        tree = build_test_tree()

        ids = [tree.R.id, tree.A.id, tree.B.id, tree.C.id,
               tree.D.id, tree.E.id, tree.F.id]
        @test length(unique(ids)) == 7

        @test tree.R.depth == 0
        @test tree.A.depth == 1
        @test tree.B.depth == 1
        @test tree.C.depth == 2
        @test tree.D.depth == 2
        @test tree.E.depth == 2
        @test tree.F.depth == 3

        @test is_root(tree.R)
        @test tree.A.parent === tree.R
        @test tree.B.parent === tree.R
        @test tree.C.parent === tree.A
        @test tree.D.parent === tree.A
        @test tree.E.parent === tree.B
        @test tree.F.parent === tree.D

        @test tree.R.is_active == true
        @test tree.A.is_active == false
        @test tree.B.is_active == false
    end
end

function test_treesearch_node_find_common_ancestor()
    @testset "[treesearch] find_common_ancestor" begin
        tree = build_test_tree()

        # Activate the path R → A → D → F (simulating we're at F)
        tree.A.is_active = true
        tree.D.is_active = true
        tree.F.is_active = true

        @test find_common_ancestor(tree.F, tree.C) === tree.A
        @test find_common_ancestor(tree.F, tree.E) === tree.R
        @test find_common_ancestor(tree.F, tree.D) === tree.D
        @test find_common_ancestor(tree.F, tree.F) === tree.F
        @test find_common_ancestor(tree.F, tree.B) === tree.R
        @test find_common_ancestor(tree.F, tree.R) === tree.R
    end
end

function test_treesearch_node_transition_accumulator()
    @testset "[treesearch] transition accumulator model" begin
        tree = build_test_tree()
        model_value = Ref(0)

        apply_fwd! = (diff::TestDiff) -> (model_value[] += diff.delta)
        apply_bwd! = (diff::TestDiff) -> (model_value[] += diff.delta)

        @test model_value[] == 0

        transition_to!(tree.R, tree.A, apply_fwd!, apply_bwd!)
        @test model_value[] == 1

        transition_to!(tree.A, tree.C, apply_fwd!, apply_bwd!)
        @test model_value[] == 11

        # C → D: backward C (-10), forward D (+20); ancestor = A. 11 + (-10) + 20 = 21
        transition_to!(tree.C, tree.D, apply_fwd!, apply_bwd!)
        @test model_value[] == 21

        # D → E: backward D (-20), backward A (-1), forward B (+2), forward E (+30)
        # ancestor = R. 21 + (-20) + (-1) + 2 + 30 = 32
        transition_to!(tree.D, tree.E, apply_fwd!, apply_bwd!)
        @test model_value[] == 32

        # E → F: backward E (-30), backward B (-2), forward A (+1), forward D (+20), forward F (+100)
        # ancestor = R. 32 + (-30) + (-2) + 1 + 20 + 100 = 121
        transition_to!(tree.E, tree.F, apply_fwd!, apply_bwd!)
        @test model_value[] == 121

        # F → R: backward F (-100), backward D (-20), backward A (-1)
        # ancestor = R. 121 + (-100) + (-20) + (-1) = 0
        transition_to!(tree.F, tree.R, apply_fwd!, apply_bwd!)
        @test model_value[] == 0
    end
end

function test_treesearch_node_active_flags()
    @testset "[treesearch] transition active flags" begin
        tree = build_test_tree()
        model_value = Ref(0)
        apply_fwd! = (diff::TestDiff) -> (model_value[] += diff.delta)
        apply_bwd! = (diff::TestDiff) -> (model_value[] += diff.delta)

        @test tree.R.is_active == true
        @test tree.A.is_active == false

        transition_to!(tree.R, tree.A, apply_fwd!, apply_bwd!)
        @test tree.R.is_active == true
        @test tree.A.is_active == true

        transition_to!(tree.A, tree.C, apply_fwd!, apply_bwd!)
        @test tree.R.is_active == true
        @test tree.A.is_active == true
        @test tree.C.is_active == true

        transition_to!(tree.C, tree.E, apply_fwd!, apply_bwd!)
        @test tree.R.is_active == true
        @test tree.A.is_active == false
        @test tree.C.is_active == false
        @test tree.B.is_active == true
        @test tree.E.is_active == true

        transition_to!(tree.E, tree.R, apply_fwd!, apply_bwd!)
        @test tree.R.is_active == true
        @test tree.B.is_active == false
        @test tree.E.is_active == false
    end
end

function test_treesearch_node_self_transition()
    @testset "[treesearch] transition self" begin
        tree = build_test_tree()
        model_value = Ref(42)
        call_count = Ref(0)
        apply_fwd! = (diff::TestDiff) -> (call_count[] += 1; model_value[] += diff.delta)
        apply_bwd! = (diff::TestDiff) -> (call_count[] += 1; model_value[] += diff.delta)

        transition_to!(tree.R, tree.R, apply_fwd!, apply_bwd!)
        @test model_value[] == 42
        @test call_count[] == 0

        tree.A.is_active = true
        transition_to!(tree.A, tree.A, apply_fwd!, apply_bwd!)
        @test model_value[] == 42
        @test call_count[] == 0
    end
end

function test_treesearch_node_collect_path()
    @testset "[treesearch] collect_path_from_ancestor" begin
        tree = build_test_tree()

        path = collect_path_from_ancestor(tree.R, tree.F)
        @test length(path) == 3
        @test path[1] === tree.A
        @test path[2] === tree.D
        @test path[3] === tree.F

        path = collect_path_from_ancestor(tree.A, tree.C)
        @test length(path) == 1
        @test path[1] === tree.C

        path = collect_path_from_ancestor(tree.R, tree.R)
        @test isempty(path)
    end
end

function test_treesearch_node()
    test_treesearch_node_construction()
    test_treesearch_node_find_common_ancestor()
    test_treesearch_node_transition_accumulator()
    test_treesearch_node_active_flags()
    test_treesearch_node_self_transition()
    test_treesearch_node_collect_path()
end
