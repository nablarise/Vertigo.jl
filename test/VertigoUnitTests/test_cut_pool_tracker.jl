# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────────────

"""Count SAF constraints across GreaterThan, LessThan, and EqualTo sets."""
function count_linear_constraints(backend)
    F = MOI.ScalarAffineFunction{Float64}
    n = 0
    for S in (MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.EqualTo{Float64})
        n += MOI.get(backend, MOI.NumberOfConstraints{F, S}())
    end
    return n
end

"""
Build a local diff pair that activates `cuts` (fwd=activate, bwd=deactivate).
"""
function _local_activate_diff(cuts::GlobalCut...)
    fwd = GlobalCutPoolDiff(
        Dict(c.id => ActivateGlobalCutChange(c) for c in cuts),
        Dict{Int, DeactivateGlobalCutChange}(),
    )
    bwd = GlobalCutPoolDiff(
        Dict{Int, ActivateGlobalCutChange}(),
        Dict(c.id => DeactivateGlobalCutChange(c) for c in cuts),
    )
    return (fwd, bwd)
end

"""
Build a local diff pair that deactivates `cuts` (fwd=deactivate, bwd=activate).
"""
function _local_deactivate_diff(cuts::GlobalCut...)
    fwd = GlobalCutPoolDiff(
        Dict{Int, ActivateGlobalCutChange}(),
        Dict(c.id => DeactivateGlobalCutChange(c) for c in cuts),
    )
    bwd = GlobalCutPoolDiff(
        Dict(c.id => ActivateGlobalCutChange(c) for c in cuts),
        Dict{Int, DeactivateGlobalCutChange}(),
    )
    return (fwd, bwd)
end

"""
Verify the backend and helper match the expected active cut set.

Checks: count of SAF constraints, set of active cut IDs in helper,
constraint function coefficients, and concrete set type for each expected cut.
"""
function _verify_gcp_node_state(backend, helper, expected_ids, cuts_by_id)
    @test length(helper.active_cuts) == length(expected_ids)
    @test count_linear_constraints(backend) == length(expected_ids)
    @test Set(keys(helper.active_cuts)) == Set(expected_ids)
    for id in expected_ids
        cut = cuts_by_id[id]
        ci = helper.active_cuts[id]
        f = MOI.get(backend, MOI.ConstraintFunction(), ci)
        s = MOI.get(backend, MOI.ConstraintSet(), ci)
        @test typeof(s) == typeof(cut.set)
        @test s == cut.set
        actual = Dict(t.variable => t.coefficient for t in f.terms)
        for term in cut.terms
            @test haskey(actual, term.variable)
            @test actual[term.variable] ≈ term.coefficient
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Test 1 — 7-node tree with mixed set types, exhaustive permutations
# ────────────────────────────────────────────────────────────────────────────────────────
#
# Tree topology:
#       | -> 7
#  1 -> 2 -> 3
#  |--> 4 -> 5 -> 6
#
# Active cuts per node:
#   1: {c8}
#   2: {c8, c1, c11}
#   3: {c8, c1, c11, c3, c7}
#   4: {c8, c4}
#   5: {c8, c4, c5, c10}
#   6: {c8, c4, c5, c10, c6}
#   7: {c8, c1, c11, c2, c9}
#
# Cut sets: mix of GreaterThan, LessThan, EqualTo.

function test_cut_pool_tracker_tree7()
    @testset "[global_cut_pool_tracker] 7-node tree (mixed set types)" begin
        m = MOI.instantiate(HiGHS.Optimizer)
        MOI.set(m, MOI.Silent(), true)
        x, y, z = MOI.add_variables(m, 3)
        MOI.add_constraint(m, x, MOI.GreaterThan(0.0))
        MOI.add_constraint(m, y, MOI.GreaterThan(0.0))
        MOI.add_constraint(m, z, MOI.GreaterThan(0.0))
        obj = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y),
             MOI.ScalarAffineTerm(1.0, z)],
            0.0,
        )
        MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
        MOI.set(m, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        T(v, c) = MOI.ScalarAffineTerm(c, v)
        tracker = GlobalCutPoolTracker()

        c1  = register_cut!(tracker, [T(x,3.),T(y,3.),T(z,2.)], MOI.GreaterThan(1.0))
        c2  = register_cut!(tracker, [T(x,3.),T(y,2.),T(z,2.)], MOI.GreaterThan(1.0))
        c3  = register_cut!(tracker, [T(x,2.),T(y,2.),T(z,2.)], MOI.LessThan(10.0))
        c4  = register_cut!(tracker, [T(x,2.),T(y,2.),T(z,1.)], MOI.EqualTo(3.0))
        c5  = register_cut!(tracker, [T(x,1.),T(y,2.),T(z,2.)], MOI.GreaterThan(1.0))
        c6  = register_cut!(tracker, [T(x,1.),T(y,1.),T(z,2.)], MOI.LessThan(8.0))
        c7  = register_cut!(tracker, [T(x,1.),T(y,1.),T(z,1.)], MOI.GreaterThan(1.0))
        c8  = register_cut!(tracker, [T(x,4.),T(y,3.),T(z,4.)], MOI.GreaterThan(1.0))
        c9  = register_cut!(tracker, [T(x,3.),T(y,4.),T(z,3.)], MOI.LessThan(15.0))
        c10 = register_cut!(tracker, [T(x,4.),T(y,4.),T(z,4.)], MOI.EqualTo(12.0))
        c11 = register_cut!(tracker, [T(x,5.),T(y,4.),T(z,5.)], MOI.GreaterThan(1.0))

        cuts_by_id = Dict(c.id => c for c in [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11])
        helper = transform_model!(tracker, m)

        state_root = root_state(tracker, m)

        # Node 1 — add c8
        (lf1, lb1) = _local_activate_diff(c8)
        fwd1 = merge_forward_change_diff(forward(state_root), lf1)
        bwd1 = merge_backward_change_diff(backward(state_root), lb1)
        state1 = new_state(tracker, fwd1, bwd1)

        # Node 2 — child of 1, add c1 and c11
        (lf2, lb2) = _local_activate_diff(c1, c11)
        fwd2 = merge_forward_change_diff(fwd1, lf2)
        bwd2 = merge_backward_change_diff(bwd1, lb2)
        state2 = new_state(tracker, fwd2, bwd2)

        # Node 3 — child of 2, add c3 and c7
        (lf3, lb3) = _local_activate_diff(c3, c7)
        fwd3 = merge_forward_change_diff(fwd2, lf3)
        bwd3 = merge_backward_change_diff(bwd2, lb3)
        state3 = new_state(tracker, fwd3, bwd3)

        # Node 4 — child of 1, add c4
        (lf4, lb4) = _local_activate_diff(c4)
        fwd4 = merge_forward_change_diff(fwd1, lf4)
        bwd4 = merge_backward_change_diff(bwd1, lb4)
        state4 = new_state(tracker, fwd4, bwd4)

        # Node 5 — child of 4, add c5 and c10
        (lf5, lb5) = _local_activate_diff(c5, c10)
        fwd5 = merge_forward_change_diff(fwd4, lf5)
        bwd5 = merge_backward_change_diff(bwd4, lb5)
        state5 = new_state(tracker, fwd5, bwd5)

        # Node 6 — child of 5, add c6
        (lf6, lb6) = _local_activate_diff(c6)
        fwd6 = merge_forward_change_diff(fwd5, lf6)
        bwd6 = merge_backward_change_diff(bwd5, lb6)
        state6 = new_state(tracker, fwd6, bwd6)

        # Node 7 — child of 2, add c2 and c9
        (lf7, lb7) = _local_activate_diff(c2, c9)
        fwd7 = merge_forward_change_diff(fwd2, lf7)
        bwd7 = merge_backward_change_diff(bwd2, lb7)
        state7 = new_state(tracker, fwd7, bwd7)

        states = [state1, state2, state3, state4, state5, state6, state7]
        expected = [
            [c8.id],
            [c8.id, c1.id, c11.id],
            [c8.id, c1.id, c11.id, c3.id, c7.id],
            [c8.id, c4.id],
            [c8.id, c4.id, c5.id, c10.id],
            [c8.id, c4.id, c5.id, c10.id, c6.id],
            [c8.id, c1.id, c11.id, c2.id, c9.id],
        ]

        current = state_root
        for perm in _all_permutations(7)
            for idx in perm
                next = states[idx]
                recover_state!(m, current, next, helper)
                _verify_gcp_node_state(m, helper, expected[idx], cuts_by_id)
                current = next
            end
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Test 2 — deactivation at a sibling node
# ────────────────────────────────────────────────────────────────────────────────────────
#
#     1 (activate c_global, GreaterThan)
#    / \
#   2   3 (deactivate c_global)
#
# Expected: nodes 1 and 2 have c_global active, node 3 has 0 active cuts.

function test_cut_pool_tracker_deactivation()
    @testset "[global_cut_pool_tracker] deactivation at sibling node" begin
        m = MOI.instantiate(HiGHS.Optimizer)
        MOI.set(m, MOI.Silent(), true)
        x = MOI.add_variable(m)
        MOI.add_constraint(m, x, MOI.GreaterThan(0.0))
        obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
        MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
        MOI.set(m, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        tracker = GlobalCutPoolTracker()
        c_global = register_cut!(
            tracker,
            [MOI.ScalarAffineTerm(2.0, x)],
            MOI.GreaterThan(0.5),
        )
        cuts_by_id = Dict(c_global.id => c_global)
        helper = transform_model!(tracker, m)

        state_root = root_state(tracker, m)

        # Node 1 activates c_global
        (lf1, lb1) = _local_activate_diff(c_global)
        fwd1 = merge_forward_change_diff(forward(state_root), lf1)
        bwd1 = merge_backward_change_diff(backward(state_root), lb1)
        state1 = new_state(tracker, fwd1, bwd1)

        # Node 2 — child of 1, no local change
        state2 = new_state(tracker, fwd1, bwd1)

        # Node 3 — child of 1, deactivates c_global
        (lf3, lb3) = _local_deactivate_diff(c_global)
        fwd3 = merge_forward_change_diff(fwd1, lf3)
        bwd3 = merge_backward_change_diff(bwd1, lb3)
        state3 = new_state(tracker, fwd3, bwd3)

        states = [state1, state2, state3]
        expected = [[c_global.id], [c_global.id], Int[]]

        current = state_root
        for perm in _all_permutations(3)
            for idx in perm
                next = states[idx]
                recover_state!(m, current, next, helper)
                _verify_gcp_node_state(m, helper, expected[idx], cuts_by_id)
                current = next
            end
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Test 3 — same cut deactivated then reactivated on a linear path
# ────────────────────────────────────────────────────────────────────────────────────────
#
#   1 (activate c_global, LessThan)
#   |
#   2 (deactivate c_global)
#   |
#   3 (reactivate c_global)
#
# Expected: node 1: {c_global}, node 2: {}, node 3: {c_global}

function test_cut_pool_tracker_reactivation()
    @testset "[global_cut_pool_tracker] deactivate then reactivate on linear path" begin
        m = MOI.instantiate(HiGHS.Optimizer)
        MOI.set(m, MOI.Silent(), true)
        x = MOI.add_variable(m)
        MOI.add_constraint(m, x, MOI.GreaterThan(0.0))
        obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
        MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
        MOI.set(m, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        tracker = GlobalCutPoolTracker()
        c_global = register_cut!(
            tracker,
            [MOI.ScalarAffineTerm(3.0, x)],
            MOI.LessThan(9.0),
        )
        cuts_by_id = Dict(c_global.id => c_global)
        helper = transform_model!(tracker, m)

        state_root = root_state(tracker, m)

        # Node 1 activates c_global
        (lf1, lb1) = _local_activate_diff(c_global)
        fwd1 = merge_forward_change_diff(forward(state_root), lf1)
        bwd1 = merge_backward_change_diff(backward(state_root), lb1)
        state1 = new_state(tracker, fwd1, bwd1)

        # Node 2 — child of 1, deactivates c_global
        (lf2, lb2) = _local_deactivate_diff(c_global)
        fwd2 = merge_forward_change_diff(fwd1, lf2)
        bwd2 = merge_backward_change_diff(bwd1, lb2)
        state2 = new_state(tracker, fwd2, bwd2)

        # Node 3 — child of 2, reactivates c_global
        (lf3, lb3) = _local_activate_diff(c_global)
        fwd3 = merge_forward_change_diff(fwd2, lf3)
        bwd3 = merge_backward_change_diff(bwd2, lb3)
        state3 = new_state(tracker, fwd3, bwd3)

        states = [state1, state2, state3]
        expected = [[c_global.id], Int[], [c_global.id]]

        current = state_root
        for perm in _all_permutations(3)
            for idx in perm
                next = states[idx]
                recover_state!(m, current, next, helper)
                _verify_gcp_node_state(m, helper, expected[idx], cuts_by_id)
                current = next
            end
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function test_cut_pool_tracker()
    test_cut_pool_tracker_tree7()
    test_cut_pool_tracker_deactivation()
    test_cut_pool_tracker_reactivation()
end
