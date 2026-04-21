# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Helper: read a variable's objective coefficient from the MOI model.
function _obj_coeff(model, var::MOI.VariableIndex)
    obj = MOI.get(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
    )
    for term in obj.terms
        term.variable == var && return term.coefficient
    end
    return 0.0
end

"""
    build_phase_test_context()

Build a minimal ColGenWorkspace for testing phase transitions.

Master (MOI, minimization):
  min  5 λ₁ + 8 λ₂ + 3.5 y_cont + 7 y_int
  s.t. c1:  λ₁ + λ₂ + y_cont + y_int == 1   (EqualTo)
       c2:  λ₁ + λ₂                  <= 2   (LessThan, convexity UB)
       c3:                             >= 0   (GreaterThan, convexity LB)
       0 <= y_cont <= 10   (continuous)
       0 <= y_int  <= 5    (integer)

λ₁, λ₂ are column variables registered in the pool with
costs 5.0 and 8.0. y_cont, y_int are pure master variables.
No actual subproblem models are needed.
"""
function build_phase_test_context()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(
            MOI.Utilities.Model{Float64}()
        ),
        HiGHS.Optimizer()
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # Variables: λ₁, λ₂, y_cont, y_int
    λ1 = MOI.add_variable(model)
    λ2 = MOI.add_variable(model)
    y_cont = MOI.add_variable(model)
    y_int = MOI.add_variable(model)

    # Bounds
    for v in (λ1, λ2)
        MOI.add_constraint(model, v, MOI.GreaterThan(0.0))
    end
    MOI.add_constraint(model, y_cont, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y_cont, MOI.LessThan(10.0))
    MOI.add_constraint(model, y_int, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y_int, MOI.LessThan(5.0))
    MOI.add_constraint(model, y_int, MOI.Integer())

    # c1: λ₁ + λ₂ + y_cont + y_int == 1
    c1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, λ1),
             MOI.ScalarAffineTerm(1.0, λ2),
             MOI.ScalarAffineTerm(1.0, y_cont),
             MOI.ScalarAffineTerm(1.0, y_int)],
            0.0
        ),
        MOI.EqualTo(1.0)
    )

    # c2: λ₁ + λ₂ <= 2 (convexity UB)
    c2 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, λ1),
             MOI.ScalarAffineTerm(1.0, λ2)],
            0.0
        ),
        MOI.LessThan(2.0)
    )

    # c3: 0 >= 0 (convexity LB)
    c3 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm{Float64}[], 0.0
        ),
        MOI.GreaterThan(0.0)
    )

    # Objective: 5 λ₁ + 8 λ₂ + 3.5 y_cont + 7 y_int
    MOI.set(model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(5.0, λ1),
             MOI.ScalarAffineTerm(8.0, λ2),
             MOI.ScalarAffineTerm(3.5, y_cont),
             MOI.ScalarAffineTerm(7.0, y_int)],
            0.0
        )
    )

    # Decomposition with pure master variables (no real subproblem).
    SpVar = MOI.VariableIndex
    CstrEq = typeof(c1)
    builder = DWReformulationBuilder{Nothing}(minimize=true)

    add_subproblem!(builder, PricingSubproblemId(1), 0.0, 0.0, 2.0)
    add_coupling_constraint!(builder, c1, 1.0)

    add_pure_master_variable!(
        builder, y_cont, 3.5, 0.0, 10.0, false
    )
    add_pure_master_coupling!(builder, y_cont, c1, 1.0)

    add_pure_master_variable!(
        builder, y_int, 7.0, 0.0, 5.0, true
    )
    add_pure_master_coupling!(builder, y_int, c1, 1.0)

    decomp = build(builder)

    # Column pool with two columns.
    pool = ColumnPool()
    sp_var_dummy = MOI.VariableIndex(9999)
    sp1 = PricingSubproblemId(1)
    sol1 = Vertigo.Reformulation._SpSolution(sp1, 5.0, [(sp_var_dummy, 1.0)])
    sol2 = Vertigo.Reformulation._SpSolution(sp1, 8.0, [(sp_var_dummy, 1.0)])
    record_column!(pool, λ1, sp1, sol1, 5.0)
    record_column!(pool, λ2, sp1, sol2, 8.0)

    conv_ub = Dict{PricingSubproblemId,TaggedCI}(sp1 => TaggedCI(c2))
    conv_lb = Dict{PricingSubproblemId,TaggedCI}(sp1 => TaggedCI(c3))

    set_models!(decomp, model, Dict{PricingSubproblemId,Any}(), conv_ub, conv_lb)

    config = ColGenConfig()
    ctx = ColGenWorkspace(decomp, pool,
        Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        config
    )

    return ctx, (λ1=λ1, λ2=λ2, y_cont=y_cont, y_int=y_int)
end

# ──────────────────────────────────────────────────────────────────

function test_setup_reformulation_phase0_relaxes_integrality()
    @testset "[setup_reformulation] phase 0 relaxes integrality" begin
        ctx, _ = build_phase_test_context()
        model = master_model(ctx.decomp)

        int_cis = MOI.get(model,
            MOI.ListOfConstraintIndices{
                MOI.VariableIndex,MOI.Integer
            }())
        @test length(int_cis) == 1

        Vertigo.ColGen.setup_reformulation!(ctx, Phase0())

        int_cis2 = MOI.get(model,
            MOI.ListOfConstraintIndices{
                MOI.VariableIndex,MOI.Integer
            }())
        @test isempty(int_cis2)

        # Artificial variables were created for c1 (EqualTo)
        # and convexity constraints.
        @test !isempty(ctx.eq_art_vars)
    end
end

function test_setup_reformulation_phase1_zeroes_costs()
    @testset "[setup_reformulation] phase 1 zeroes all variable costs" begin
        ctx, vars = build_phase_test_context()
        model = master_model(ctx.decomp)

        Vertigo.ColGen.setup_reformulation!(ctx, Phase0())

        # Original costs still present after phase 0.
        @test _obj_coeff(model, vars.λ1) ≈ 5.0
        @test _obj_coeff(model, vars.λ2) ≈ 8.0
        @test _obj_coeff(model, vars.y_cont) ≈ 3.5
        @test _obj_coeff(model, vars.y_int) ≈ 7.0

        Vertigo.ColGen.setup_reformulation!(ctx, Phase1())

        # All original costs zeroed.
        @test _obj_coeff(model, vars.λ1) ≈ 0.0
        @test _obj_coeff(model, vars.λ2) ≈ 0.0
        @test _obj_coeff(model, vars.y_cont) ≈ 0.0
        @test _obj_coeff(model, vars.y_int) ≈ 0.0
    end
end

function test_setup_reformulation_phase2_restores_costs()
    @testset "[setup_reformulation] phase 2 restores original costs" begin
        ctx, vars = build_phase_test_context()
        model = master_model(ctx.decomp)

        Vertigo.ColGen.setup_reformulation!(ctx, Phase0())
        Vertigo.ColGen.setup_reformulation!(ctx, Phase1())

        @test _obj_coeff(model, vars.y_cont) ≈ 0.0
        @test _obj_coeff(model, vars.y_int) ≈ 0.0

        Vertigo.ColGen.setup_reformulation!(ctx, Phase2())

        # Column costs restored from pool.
        @test _obj_coeff(model, vars.λ1) ≈ 5.0
        @test _obj_coeff(model, vars.λ2) ≈ 8.0
        # Pure master variable costs restored from decomposition.
        @test _obj_coeff(model, vars.y_cont) ≈ 3.5
        @test _obj_coeff(model, vars.y_int) ≈ 7.0
    end
end

function test_setup_reformulation_phase2_removes_art_vars()
    @testset "[setup_reformulation] phase 2 removes artificial variables" begin
        ctx, _ = build_phase_test_context()

        Vertigo.ColGen.setup_reformulation!(ctx, Phase0())

        n_art = length(ctx.eq_art_vars) +
                length(ctx.leq_art_vars) +
                length(ctx.geq_art_vars)
        @test n_art > 0

        Vertigo.ColGen.setup_reformulation!(ctx, Phase1())
        Vertigo.ColGen.setup_reformulation!(ctx, Phase2())

        @test isempty(ctx.eq_art_vars)
        @test isempty(ctx.leq_art_vars)
        @test isempty(ctx.geq_art_vars)
    end
end

function test_setup_reformulation_full_sequence()
    @testset "[setup_reformulation] full phase 0 → 1 → 2 roundtrip" begin
        ctx, vars = build_phase_test_context()
        model = master_model(ctx.decomp)

        @test _obj_coeff(model, vars.λ1) ≈ 5.0
        @test _obj_coeff(model, vars.λ2) ≈ 8.0
        @test _obj_coeff(model, vars.y_cont) ≈ 3.5
        @test _obj_coeff(model, vars.y_int) ≈ 7.0

        Vertigo.ColGen.setup_reformulation!(ctx, Phase0())

        @test _obj_coeff(model, vars.λ1) ≈ 5.0
        @test _obj_coeff(model, vars.λ2) ≈ 8.0
        @test _obj_coeff(model, vars.y_cont) ≈ 3.5
        @test _obj_coeff(model, vars.y_int) ≈ 7.0

        Vertigo.ColGen.setup_reformulation!(ctx, Phase1())

        @test _obj_coeff(model, vars.λ1) ≈ 0.0
        @test _obj_coeff(model, vars.λ2) ≈ 0.0
        @test _obj_coeff(model, vars.y_cont) ≈ 0.0
        @test _obj_coeff(model, vars.y_int) ≈ 0.0

        Vertigo.ColGen.setup_reformulation!(ctx, Phase2())

        # Model should still solve.
        MOI.optimize!(model)
        status = MOI.get(model, MOI.TerminationStatus())
        @test status == MOI.OPTIMAL

        # All costs intact after full roundtrip.
        @test _obj_coeff(model, vars.λ1) ≈ 5.0
        @test _obj_coeff(model, vars.λ2) ≈ 8.0
        @test _obj_coeff(model, vars.y_cont) ≈ 3.5
        @test _obj_coeff(model, vars.y_int) ≈ 7.0
    end
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

function test_setup_reformulation()
    test_setup_reformulation_phase0_relaxes_integrality()
    test_setup_reformulation_phase1_zeroes_costs()
    test_setup_reformulation_phase2_restores_costs()
    test_setup_reformulation_phase2_removes_art_vars()
    test_setup_reformulation_full_sequence()
end
