# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# Test 1 — add/remove basic column
# ────────────────────────────────────────────────────────────────────────────────────────
#
# Model: min x, x >= 0
#   c1: x >= 1   (GreaterThan SAF)
#   c2: x <= 100 (LessThan SAF)
#
# Build a column z with obj=3.0, lb=0, ub=10, entries in c1 (coeff=2.0) and c2 (coeff=1.0).
# Forward: add z → verify 2 variables, coefficients in objective and constraints.
# Backward: remove z → verify 1 variable, z no longer valid.

function _build_ct_backend()
    m = MOI.instantiate(HiGHS.Optimizer)
    MOI.set(m, MOI.Silent(), true)
    x = MOI.add_variable(m)
    MOI.add_constraint(m, x, MOI.GreaterThan(0.0))
    obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
    MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
    MOI.set(m, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    c1 = MOI.add_constraint(
        m,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
        MOI.GreaterThan(1.0),
    )
    c2 = MOI.add_constraint(
        m,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
        MOI.LessThan(100.0),
    )
    return m, x, c1, c2
end

function test_column_tracker_add_remove_basic()
    @testset "[column_tracker] add/remove basic column" begin
        m, x, c1, c2 = _build_ct_backend()
        t = ColumnTracker()
        helper = transform_model!(t, m)

        z_data = ColumnData(
            next_id!(t),
            3.0, 0.0, 10.0,
            [(c1, 2.0), (c2, 1.0)],
        )

        local_fwd = ColumnChangeDiff([AddColumnChange(z_data)], RemoveColumnChange[])
        local_bwd = ColumnChangeDiff(AddColumnChange[], [RemoveColumnChange(z_data)])

        # Apply forward: add z
        apply_change!(m, local_fwd, helper)

        @test MOI.get(m, MOI.NumberOfVariables()) == 2
        vi_z = helper.active_columns[z_data.id]

        # Objective coefficient for z
        obj_f = MOI.get(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        obj_coeffs = Dict(t.variable => t.coefficient for t in obj_f.terms)
        @test haskey(obj_coeffs, vi_z)
        @test obj_coeffs[vi_z] ≈ 3.0

        # c1 coefficient for z
        f_c1 = MOI.get(m, MOI.ConstraintFunction(), c1)
        c1_coeffs = Dict(t.variable => t.coefficient for t in f_c1.terms)
        @test haskey(c1_coeffs, vi_z)
        @test c1_coeffs[vi_z] ≈ 2.0

        # c2 coefficient for z
        f_c2 = MOI.get(m, MOI.ConstraintFunction(), c2)
        c2_coeffs = Dict(t.variable => t.coefficient for t in f_c2.terms)
        @test haskey(c2_coeffs, vi_z)
        @test c2_coeffs[vi_z] ≈ 1.0

        # Apply backward: remove z
        apply_change!(m, local_bwd, helper)

        @test MOI.get(m, MOI.NumberOfVariables()) == 1
        @test !MOI.is_valid(m, vi_z)
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Test 2 — tree with mixed constraint types (3 nodes)
# ────────────────────────────────────────────────────────────────────────────────────────
#
# Model: min x, x >= 0
#   c_ge: 2x >= 3   (GreaterThan SAF)
#   c_le: x <= 10   (LessThan SAF)
#   c_eq: x == 5    (EqualTo SAF)
#
# Node 1: root (1 var)
# Node 2: child of 1, adds z1 (obj=1, lb=0, ub=5, entries in c_ge and c_le)
# Node 3: child of 1, adds z2 (obj=2, lb=0, ub=3, entries in c_eq and c_ge)
#
# Verify active_columns count and variable count for all permutations over {1,2,3}.

struct _CTMixedModel
    backend
    c_ge::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}
    c_le::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
    c_eq::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
end

function _build_ct_mixed_backend()
    m = MOI.instantiate(HiGHS.Optimizer)
    MOI.set(m, MOI.Silent(), true)
    x = MOI.add_variable(m)
    MOI.add_constraint(m, x, MOI.GreaterThan(0.0))
    obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
    MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
    MOI.set(m, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    c_ge = MOI.add_constraint(
        m,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(2.0, x)], 0.0),
        MOI.GreaterThan(3.0),
    )
    c_le = MOI.add_constraint(
        m,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
        MOI.LessThan(10.0),
    )
    c_eq = MOI.add_constraint(
        m,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
        MOI.EqualTo(5.0),
    )
    return _CTMixedModel(m, c_ge, c_le, c_eq)
end

function test_column_tracker_mixed_constraints()
    @testset "[column_tracker] tree with mixed constraint types (3 nodes)" begin
        mdl = _build_ct_mixed_backend()
        m = mdl.backend
        t = ColumnTracker()
        helper = transform_model!(t, m)

        z1_data = ColumnData(
            next_id!(t),
            1.0, 0.0, 5.0,
            [(mdl.c_ge, 1.5), (mdl.c_le, 0.5)],
        )
        z2_data = ColumnData(
            next_id!(t),
            2.0, 0.0, 3.0,
            [(mdl.c_eq, 1.0), (mdl.c_ge, 0.25)],
        )

        # Node 1 — root
        state1 = root_state(t, m)

        # Node 2 — child of 1, adds z1
        local_fwd2 = ColumnChangeDiff([AddColumnChange(z1_data)], RemoveColumnChange[])
        local_bwd2 = ColumnChangeDiff(AddColumnChange[], [RemoveColumnChange(z1_data)])
        fwd2 = merge_forward_change_diff(forward(state1), local_fwd2)
        bwd2 = merge_backward_change_diff(backward(state1), local_bwd2)
        state2 = new_state(t, fwd2, bwd2)

        # Node 3 — child of 1, adds z2
        local_fwd3 = ColumnChangeDiff([AddColumnChange(z2_data)], RemoveColumnChange[])
        local_bwd3 = ColumnChangeDiff(AddColumnChange[], [RemoveColumnChange(z2_data)])
        fwd3 = merge_forward_change_diff(forward(state1), local_fwd3)
        bwd3 = merge_backward_change_diff(backward(state1), local_bwd3)
        state3 = new_state(t, fwd3, bwd3)

        states = [state1, state2, state3]
        # (expected active_columns count, expected var count in model)
        expected = [(0, 1), (1, 2), (1, 2)]

        current_state = state1
        for perm in _all_permutations(3)
            for idx in perm
                next = states[idx]
                recover_state!(m, current_state, next, helper)
                (exp_cols, exp_vars) = expected[idx]
                @test length(helper.active_columns) == exp_cols
                @test MOI.get(m, MOI.NumberOfVariables()) == exp_vars
                current_state = next
            end
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Test 3 — deep tree column inheritance (4 nodes)
# ────────────────────────────────────────────────────────────────────────────────────────
#
# Node 1: root, 1 var (x)
# Node 2: child of 1, adds z1 → 2 vars
# Node 3: child of 2, adds z2 → 3 vars
# Node 4: child of 2, adds z3 → 3 vars (sibling of 3)
#
# Verify variable counts for all permutations over {1,2,3,4}.

function test_column_tracker_deep_tree()
    @testset "[column_tracker] deep tree column inheritance (4 nodes)" begin
        m = MOI.instantiate(HiGHS.Optimizer)
        MOI.set(m, MOI.Silent(), true)
        x = MOI.add_variable(m)
        MOI.add_constraint(m, x, MOI.GreaterThan(0.0))
        c1 = MOI.add_constraint(
            m,
            MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
            MOI.GreaterThan(0.0),
        )
        obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
        MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
        MOI.set(m, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        t = ColumnTracker()
        helper = transform_model!(t, m)

        z1_data = ColumnData(next_id!(t), 1.0, 0.0, Inf, [(c1, 1.0)])
        z2_data = ColumnData(next_id!(t), 2.0, 0.0, Inf, [(c1, 1.0)])
        z3_data = ColumnData(next_id!(t), 3.0, 0.0, Inf, [(c1, 1.0)])

        # Node 1 — root
        state1 = root_state(t, m)

        # Node 2 — adds z1
        local_fwd2 = ColumnChangeDiff([AddColumnChange(z1_data)], RemoveColumnChange[])
        local_bwd2 = ColumnChangeDiff(AddColumnChange[], [RemoveColumnChange(z1_data)])
        fwd2 = merge_forward_change_diff(forward(state1), local_fwd2)
        bwd2 = merge_backward_change_diff(backward(state1), local_bwd2)
        state2 = new_state(t, fwd2, bwd2)

        # Node 3 — child of 2, adds z2
        local_fwd3 = ColumnChangeDiff([AddColumnChange(z2_data)], RemoveColumnChange[])
        local_bwd3 = ColumnChangeDiff(AddColumnChange[], [RemoveColumnChange(z2_data)])
        fwd3 = merge_forward_change_diff(forward(state2), local_fwd3)
        bwd3 = merge_backward_change_diff(backward(state2), local_bwd3)
        state3 = new_state(t, fwd3, bwd3)

        # Node 4 — child of 2, adds z3 (sibling of node 3)
        local_fwd4 = ColumnChangeDiff([AddColumnChange(z3_data)], RemoveColumnChange[])
        local_bwd4 = ColumnChangeDiff(AddColumnChange[], [RemoveColumnChange(z3_data)])
        fwd4 = merge_forward_change_diff(forward(state2), local_fwd4)
        bwd4 = merge_backward_change_diff(backward(state2), local_bwd4)
        state4 = new_state(t, fwd4, bwd4)

        states = [state1, state2, state3, state4]
        expected_vars = [1, 2, 3, 3]

        current_state = state1
        for perm in _all_permutations(4)
            for idx in perm
                next = states[idx]
                recover_state!(m, current_state, next, helper)
                @test MOI.get(m, MOI.NumberOfVariables()) == expected_vars[idx]
                current_state = next
            end
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function test_column_tracker()
    test_column_tracker_add_remove_basic()
    test_column_tracker_mixed_constraints()
    test_column_tracker_deep_tree()
end
