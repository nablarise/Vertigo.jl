# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────────────

"""Count active cuts via the helper (set-type-agnostic)."""
function count_active_cuts(helper::LocalCutTrackerHelper)
    return length(helper.active_cuts)
end

"""Retrieve the constraint function and set for a given cut id."""
function get_cut_constraint(backend, helper::LocalCutTrackerHelper, cut_id::Int)
    tagged = helper.active_cuts[cut_id]
    return with_typed_ci(tagged) do ci
        f = MOI.get(backend, MOI.ConstraintFunction(), ci)
        s = MOI.get(backend, MOI.ConstraintSet(), ci)
        (f, s)
    end
end

"""Build a node-level (local) diff pair for adding a single cut."""
function _local_add_diff(cut::LocalCut)
    fwd = LocalCutChangeDiff([AddLocalCutChange(cut)], [])
    bwd = LocalCutChangeDiff([], [RemoveLocalCutChange(cut)])
    return (fwd, bwd)
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Build a simple backend: min x + y + z, x,y,z >= 0
# ────────────────────────────────────────────────────────────────────────────────────────

function _build_simple_backend()
    m = MOI.instantiate(HiGHS.Optimizer)
    MOI.set(m, MOI.Silent(), true)
    x, y, z = MOI.add_variables(m, 3)
    MOI.add_constraint(m, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(m, y, MOI.GreaterThan(0.0))
    MOI.add_constraint(m, z, MOI.GreaterThan(0.0))
    obj = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y), MOI.ScalarAffineTerm(1.0, z)],
        0.0,
    )
    MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
    MOI.set(m, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return m, [x, y, z]
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Verify helper state after recover_state!
# ────────────────────────────────────────────────────────────────────────────────────────

function _verify_node_state(backend, helper, expected_count, expected_cut_ids, cuts_by_id)
    @test count_active_cuts(helper) == expected_count
    @test Set(keys(helper.active_cuts)) == Set(expected_cut_ids)
    for id in expected_cut_ids
        cut = cuts_by_id[id]
        (f, s) = get_cut_constraint(backend, helper, id)
        # Check set matches
        @test s == cut.set
        # Check each term coefficient matches (order may differ, so match by variable)
        actual_coeff = Dict(t.variable => t.coefficient for t in f.terms)
        for term in cut.terms
            @test haskey(actual_coeff, term.variable)
            @test actual_coeff[term.variable] ≈ term.coefficient atol=1e-9
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Test 1 — simple tree with local cuts (4 nodes)
# ────────────────────────────────────────────────────────────────────────────────────────
#
#      1 (root, no cut)
#     / \
#    2   3
#    |
#    4
#
# Expected active cuts:
#   node1 = {}
#   node2 = {c1}
#   node3 = {c2}
#   node4 = {c1, c3}

function test_local_cut_tracker_simple_tree()
    @testset "[local_cut_tracker] simple tree (4 nodes)" begin
        backend, vars = _build_simple_backend()
        tracker = LocalCutTracker()
        helper = transform_model!(tracker, backend)

        x, y, z = vars

        # Cuts
        c1 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)],
            MOI.GreaterThan(1.0),
        )
        c2 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(2.0, y), MOI.ScalarAffineTerm(1.0, z)],
            MOI.LessThan(5.0),
        )
        c3 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(3.0, z)],
            MOI.GreaterThan(0.5),
        )
        cuts_by_id = Dict(c1.id => c1, c2.id => c2, c3.id => c3)

        # Build node-level local diffs
        (c1_fwd, c1_bwd) = _local_add_diff(c1)
        (c2_fwd, c2_bwd) = _local_add_diff(c2)
        (c3_fwd, c3_bwd) = _local_add_diff(c3)

        # Node 1 — root, no cuts
        state1 = root_state(tracker, backend)

        # Node 2 — child of 1, adds c1
        fwd2 = merge_forward_change_diff(forward(state1), c1_fwd)
        bwd2 = merge_backward_change_diff(backward(state1), c1_bwd)
        state2 = new_state(tracker, fwd2, bwd2)

        # Node 3 — child of 1, adds c2
        fwd3 = merge_forward_change_diff(forward(state1), c2_fwd)
        bwd3 = merge_backward_change_diff(backward(state1), c2_bwd)
        state3 = new_state(tracker, fwd3, bwd3)

        # Node 4 — child of 2, adds c3
        fwd4 = merge_forward_change_diff(forward(state2), c3_fwd)
        bwd4 = merge_backward_change_diff(backward(state2), c3_bwd)
        state4 = new_state(tracker, fwd4, bwd4)

        states = [state1, state2, state3, state4]
        expected = [
            (0, Int[]),
            (1, [c1.id]),
            (1, [c2.id]),
            (2, [c1.id, c3.id]),
        ]

        # Start at root
        prev = state1
        for (next_idx, (exp_count, exp_ids)) in enumerate(expected)
            next = states[next_idx]
            recover_state!(backend, prev, next, helper)
            _verify_node_state(backend, helper, exp_count, exp_ids, cuts_by_id)
            prev = next
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Test 2 — exhaustive permutations (4-node tree)
# ────────────────────────────────────────────────────────────────────────────────────────

function test_local_cut_tracker_permutations_4()
    @testset "[local_cut_tracker] permutations exhaustive (4 nodes)" begin
        backend, vars = _build_simple_backend()
        tracker = LocalCutTracker()
        helper = transform_model!(tracker, backend)

        x, y, z = vars

        c1 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)],
            MOI.GreaterThan(1.0),
        )
        c2 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(2.0, y), MOI.ScalarAffineTerm(1.0, z)],
            MOI.LessThan(5.0),
        )
        c3 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(3.0, z)],
            MOI.GreaterThan(0.5),
        )
        cuts_by_id = Dict(c1.id => c1, c2.id => c2, c3.id => c3)

        (c1_fwd, c1_bwd) = _local_add_diff(c1)
        (c2_fwd, c2_bwd) = _local_add_diff(c2)
        (c3_fwd, c3_bwd) = _local_add_diff(c3)

        state1 = root_state(tracker, backend)

        fwd2 = merge_forward_change_diff(forward(state1), c1_fwd)
        bwd2 = merge_backward_change_diff(backward(state1), c1_bwd)
        state2 = new_state(tracker, fwd2, bwd2)

        fwd3 = merge_forward_change_diff(forward(state1), c2_fwd)
        bwd3 = merge_backward_change_diff(backward(state1), c2_bwd)
        state3 = new_state(tracker, fwd3, bwd3)

        fwd4 = merge_forward_change_diff(forward(state2), c3_fwd)
        bwd4 = merge_backward_change_diff(backward(state2), c3_bwd)
        state4 = new_state(tracker, fwd4, bwd4)

        states = [state1, state2, state3, state4]
        expected = [
            (0, Int[]),
            (1, [c1.id]),
            (1, [c2.id]),
            (2, [c1.id, c3.id]),
        ]

        _for_all_permutations(4, state1) do current_state, idx
            next = states[idx]
            recover_state!(backend, current_state, next, helper)
            (exp_count, exp_ids) = expected[idx]
            _verify_node_state(
                backend, helper, exp_count, exp_ids, cuts_by_id
            )
            return next
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Test 3 — deeper tree (5 nodes) with permutations
# ────────────────────────────────────────────────────────────────────────────────────────
#
#      1
#      |
#      2 (add c1)
#      |
#      3 (add c2)
#     / \
#    4   5
#  (+c3) (+c4)
#
# Expected:
#   node1 = {}
#   node2 = {c1}
#   node3 = {c1, c2}
#   node4 = {c1, c2, c3}
#   node5 = {c1, c2, c4}

function test_local_cut_tracker_deeper_tree()
    @testset "[local_cut_tracker] deeper tree (5 nodes) with permutations" begin
        backend, vars = _build_simple_backend()
        tracker = LocalCutTracker()
        helper = transform_model!(tracker, backend)

        x, y, z = vars

        c1 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(1.0, x)],
            MOI.GreaterThan(0.1),
        )
        c2 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(1.0, y)],
            MOI.GreaterThan(0.2),
        )
        c3 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(1.0, z)],
            MOI.LessThan(3.0),
        )
        c4 = LocalCut(
            next_id!(tracker),
            [MOI.ScalarAffineTerm(2.0, x), MOI.ScalarAffineTerm(1.0, z)],
            MOI.GreaterThan(0.5),
        )
        cuts_by_id = Dict(c1.id => c1, c2.id => c2, c3.id => c3, c4.id => c4)

        (c1_fwd, c1_bwd) = _local_add_diff(c1)
        (c2_fwd, c2_bwd) = _local_add_diff(c2)
        (c3_fwd, c3_bwd) = _local_add_diff(c3)
        (c4_fwd, c4_bwd) = _local_add_diff(c4)

        # node1 — root
        state1 = root_state(tracker, backend)

        # node2 — child of 1, adds c1
        fwd2 = merge_forward_change_diff(forward(state1), c1_fwd)
        bwd2 = merge_backward_change_diff(backward(state1), c1_bwd)
        state2 = new_state(tracker, fwd2, bwd2)

        # node3 — child of 2, adds c2
        fwd3 = merge_forward_change_diff(forward(state2), c2_fwd)
        bwd3 = merge_backward_change_diff(backward(state2), c2_bwd)
        state3 = new_state(tracker, fwd3, bwd3)

        # node4 — child of 3, adds c3
        fwd4 = merge_forward_change_diff(forward(state3), c3_fwd)
        bwd4 = merge_backward_change_diff(backward(state3), c3_bwd)
        state4 = new_state(tracker, fwd4, bwd4)

        # node5 — child of 3, adds c4
        fwd5 = merge_forward_change_diff(forward(state3), c4_fwd)
        bwd5 = merge_backward_change_diff(backward(state3), c4_bwd)
        state5 = new_state(tracker, fwd5, bwd5)

        states = [state1, state2, state3, state4, state5]
        expected = [
            (0, Int[]),
            (1, [c1.id]),
            (2, [c1.id, c2.id]),
            (3, [c1.id, c2.id, c3.id]),
            (3, [c1.id, c2.id, c4.id]),
        ]

        _for_all_permutations(5, state1) do current_state, idx
            next = states[idx]
            recover_state!(backend, current_state, next, helper)
            (exp_count, exp_ids) = expected[idx]
            _verify_node_state(
                backend, helper, exp_count, exp_ids, cuts_by_id
            )
            return next
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function test_local_cut_tracker()
    test_local_cut_tracker_simple_tree()
    test_local_cut_tracker_permutations_4()
    test_local_cut_tracker_deeper_tree()
end
