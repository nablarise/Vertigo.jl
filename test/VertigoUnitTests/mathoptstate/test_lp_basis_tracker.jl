# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────────────

# Build a simple LP: min -x - y, x,y in [0,10], x+y <= 15.
# Optimum: x=10, y=5 (or any (a,b) with a+b=15, 0<=a<=10, 0<=b<=10), obj=-15.
# Returns (backend, x, y, ci_x_ub, c1) where:
#   ci_x_ub = VariableIndex constraint for x <= 10
#   c1      = SAF constraint for x+y <= 15
function _build_lp_backend()
    m = MOI.instantiate(HiGHS.Optimizer)
    MOI.set(m, MOI.Silent(), true)
    x, y = MOI.add_variables(m, 2)
    MOI.add_constraint(m, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(m, y, MOI.GreaterThan(0.0))
    ci_x_ub = MOI.add_constraint(m, x, MOI.LessThan(10.0))
    MOI.add_constraint(m, y, MOI.LessThan(10.0))
    obj = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(-1.0, x), MOI.ScalarAffineTerm(-1.0, y)],
        0.0,
    )
    MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
    MOI.set(m, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)],
        0.0,
    )
    c1 = MOI.add_constraint(m, f, MOI.LessThan(15.0))
    return m, x, y, ci_x_ub, c1
end

# Build an LP with all three SAF constraint set types.
# min x + y, x,y >= 0, x+y >= 2, x+y <= 10, x-y == 0.
# Optimum: x=y=1, obj=2.
function _build_three_set_lp_backend()
    m = MOI.instantiate(HiGHS.Optimizer)
    MOI.set(m, MOI.Silent(), true)
    x, y = MOI.add_variables(m, 2)
    MOI.add_constraint(m, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(m, y, MOI.GreaterThan(0.0))
    obj = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)],
        0.0,
    )
    MOI.set(m, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
    MOI.set(m, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f_sum = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)],
        0.0,
    )
    f_diff = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(-1.0, y)],
        0.0,
    )
    MOI.add_constraint(m, f_sum, MOI.GreaterThan(2.0))
    MOI.add_constraint(m, f_sum, MOI.LessThan(10.0))
    MOI.add_constraint(m, f_diff, MOI.EqualTo(0.0))
    return m, x, y
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Test functions
# ────────────────────────────────────────────────────────────────────────────────────────

function test_lp_basis_tracker_capture_shape()
    @testset "[lp_basis_tracker] capture_basis returns correct shape" begin
        m, x, y, _ci_x_ub, _c1 = _build_lp_backend()
        MOI.optimize!(m)
        @test MOI.get(m, MOI.TerminationStatus()) == MOI.OPTIMAL

        basis = capture_basis(m)

        @test length(basis.var_status) == 2
        @test haskey(basis.var_status, x)
        @test haskey(basis.var_status, y)
        # Only the one SAF constraint (x+y <= 15) appears in constr_status;
        # variable bound constraints (VariableIndex constraints) are excluded.
        @test length(basis.constr_status) == 1
    end
end

function test_lp_basis_tracker_apply_noop_nothing_basis()
    @testset "[lp_basis_tracker] apply_change! no-op on nothing basis" begin
        m, _x, _y, _ci_x_ub, _c1 = _build_lp_backend()
        # Must not throw even though no solve has happened.
        @test begin
            apply_change!(m, LPBasisDiff(), nothing)
            true
        end
    end
end

function test_lp_basis_tracker_apply_restores_basis()
    @testset "[lp_basis_tracker] apply_change! restores basis after bound change" begin
        m, _x, _y, ci_x_ub, _c1 = _build_lp_backend()
        MOI.optimize!(m)
        @test MOI.get(m, MOI.TerminationStatus()) == MOI.OPTIMAL

        state_root = ModelState(LPBasisDiff(capture_basis(m)), LPBasisDiff())

        # Simulate branching: tighten x <= 8
        MOI.set(m, MOI.ConstraintSet(), ci_x_ub, MOI.LessThan(8.0))

        # Warm-start with the root basis before re-solving
        apply_change!(m, forward(state_root), nothing)

        MOI.optimize!(m)
        @test MOI.get(m, MOI.TerminationStatus()) == MOI.OPTIMAL
        # New optimum: x=8, y=7, obj = -15
        @test MOI.get(m, MOI.ObjectiveValue()) ≈ -15.0
    end
end

function test_lp_basis_tracker_update_basis()
    @testset "[lp_basis_tracker] update_basis returns new state" begin
        m, _x, _y, _ci_x_ub, _c1 = _build_lp_backend()
        MOI.optimize!(m)
        @test MOI.get(m, MOI.TerminationStatus()) == MOI.OPTIMAL

        state = root_state(LPBasisTracker(), m)
        state = update_basis(state, m)

        @test !isnothing(forward(state).basis)
        # The backward diff is carried over from the root state (empty).
        @test isnothing(backward(state).basis)
    end
end

function test_lp_basis_tracker_merge_forward_keeps_local()
    @testset "[lp_basis_tracker] merge_forward ignores parent, keeps local" begin
        parent_basis = LPBasisState(
            Dict{MOI.VariableIndex, MOI.BasisStatusCode}(),
            Dict{TaggedCI, MOI.BasisStatusCode}(),
        )
        child_basis = LPBasisState(
            Dict{MOI.VariableIndex, MOI.BasisStatusCode}(),
            Dict{TaggedCI, MOI.BasisStatusCode}(),
        )
        parent_diff = LPBasisDiff(parent_basis)
        local_diff  = LPBasisDiff(child_basis)

        result = merge_forward_change_diff(parent_diff, local_diff)

        @test result === local_diff
    end
end

function test_lp_basis_tracker_merge_backward_always_noop()
    @testset "[lp_basis_tracker] merge_backward always returns nothing diff" begin
        some_basis = LPBasisState(
            Dict{MOI.VariableIndex, MOI.BasisStatusCode}(),
            Dict{TaggedCI, MOI.BasisStatusCode}(),
        )
        d1 = LPBasisDiff(some_basis)
        d2 = LPBasisDiff(some_basis)

        result = merge_backward_change_diff(d1, d2)

        @test isnothing(result.basis)
    end
end

function test_lp_basis_tracker_robustness_stale_index()
    @testset "[lp_basis_tracker] robustness: invalid constraint index errors" begin
        m, _x, _y, _ci_x_ub, c1 = _build_lp_backend()
        MOI.optimize!(m)
        @test MOI.get(m, MOI.TerminationStatus()) == MOI.OPTIMAL

        # Capture basis (c1 is valid at this point)
        basis = capture_basis(m)
        @test haskey(basis.constr_status, TaggedCI(c1))

        # Delete the SAF constraint — c1 is now stale
        MOI.delete(m, c1)

        # Build a basis with only the stale constraint (no var statuses)
        # to avoid early return from unsupported VariableBasisStatus.
        stale_basis = LPBasisState(
            Dict{MOI.VariableIndex, MOI.BasisStatusCode}(),
            basis.constr_status,
        )

        # apply_change! must error on invalid constraint index
        @test_throws ErrorException apply_change!(
            m, LPBasisDiff(stale_basis), nothing
        )
    end
end

function test_lp_basis_tracker_three_set_types()
    @testset "[lp_basis_tracker] constr_status captures all three set types" begin
        m, _x, _y = _build_three_set_lp_backend()
        MOI.optimize!(m)
        @test MOI.get(m, MOI.TerminationStatus()) == MOI.OPTIMAL

        basis = capture_basis(m)

        F = MOI.ScalarAffineFunction{Float64}
        gt_indices = MOI.get(m, MOI.ListOfConstraintIndices{F, MOI.GreaterThan{Float64}}())
        lt_indices = MOI.get(m, MOI.ListOfConstraintIndices{F, MOI.LessThan{Float64}}())
        eq_indices = MOI.get(m, MOI.ListOfConstraintIndices{F, MOI.EqualTo{Float64}}())

        @test length(gt_indices) == 1
        @test length(lt_indices) == 1
        @test length(eq_indices) == 1

        @test haskey(basis.constr_status, TaggedCI(gt_indices[1]))
        @test haskey(basis.constr_status, TaggedCI(lt_indices[1]))
        @test haskey(basis.constr_status, TaggedCI(eq_indices[1]))
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function test_lp_basis_tracker()
    test_lp_basis_tracker_capture_shape()
    test_lp_basis_tracker_apply_noop_nothing_basis()
    test_lp_basis_tracker_apply_restores_basis()
    test_lp_basis_tracker_update_basis()
    test_lp_basis_tracker_merge_forward_keeps_local()
    test_lp_basis_tracker_merge_backward_always_noop()
    test_lp_basis_tracker_robustness_stale_index()
    test_lp_basis_tracker_three_set_types()
end
