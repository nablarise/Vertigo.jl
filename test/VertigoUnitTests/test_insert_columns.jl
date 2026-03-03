# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Helper: read a variable's coefficient in a constraint.
function _constr_coeff(model, ci, var::MOI.VariableIndex)
    f = MOI.get(model, MOI.ConstraintFunction(), ci)
    for term in f.terms
        term.variable == var && return term.coefficient
    end
    return 0.0
end

"""
    build_insert_columns_context()

Build a minimal ColGenContext for testing `insert_columns!` (Phase 0).

Master (MOI, minimization):
  min  0
  s.t. c1:  0 == 1   (coupling, EqualTo)
       c2:  0 == 1   (coupling, EqualTo)
       conv_ub:  0 <= 1   (convexity UB)
       conv_lb:  0 >= 0   (convexity LB)
       br:  0 <= 1   (branching constraint on original var x₁)

One subproblem (id = 1) with two SP variables (z1, z2):
  - z1: original cost 3.0, coupling c1 → 2.0, c2 → 1.0, maps to x₁
  - z2: original cost 5.0, coupling c1 → 0.0, c2 → 4.0, maps to x₂

Convexity bounds: lb = 0.0, ub = 1.0.
One active branching constraint on x₁.
No cuts.
"""
function build_insert_columns_context()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(
            MOI.Utilities.Model{Float64}()
        ),
        HiGHS.Optimizer()
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 0.0)
    )

    # Coupling constraints
    c1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm{Float64}[], 0.0
        ),
        MOI.EqualTo(1.0)
    )
    c2 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm{Float64}[], 0.0
        ),
        MOI.EqualTo(1.0)
    )

    # Convexity constraints
    conv_ub = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm{Float64}[], 0.0
        ),
        MOI.LessThan(1.0)
    )
    conv_lb = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm{Float64}[], 0.0
        ),
        MOI.GreaterThan(0.0)
    )

    # Branching constraint on original variable x₁
    br = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm{Float64}[], 0.0
        ),
        MOI.LessThan(1.0)
    )

    # Decomposition
    SpVar = MOI.VariableIndex
    CstrEq = typeof(c1)
    # X = Tuple{Int,Int} for original variable mappings
    OrigVar = Tuple{Int,Int}

    builder = DecompositionBuilder{
        Int,SpVar,OrigVar,CstrEq,Nothing
    }(minimize=true)

    add_subproblem!(builder, 1, 0.0, 0.0, 1.0)

    z1 = MOI.VariableIndex(1)
    z2 = MOI.VariableIndex(2)

    add_sp_variable!(builder, 1, z1, 3.0)
    add_coupling_coefficient!(builder, 1, z1, c1, 2.0)
    add_coupling_coefficient!(builder, 1, z1, c2, 1.0)
    add_mapping!(builder, (1, 1), 1, z1)   # z1 → x₁

    add_sp_variable!(builder, 1, z2, 5.0)
    add_coupling_coefficient!(builder, 1, z2, c2, 4.0)
    add_mapping!(builder, (1, 2), 1, z2)   # z2 → x₂

    add_coupling_constraint!(builder, c1, EQUAL_TO, 1.0)
    add_coupling_constraint!(builder, c2, EQUAL_TO, 1.0)

    decomp = build(builder)

    pool = ColumnPool{MOI.VariableIndex,Int,SpVar}()

    conv_ub_map = Dict{Any,Any}(1 => conv_ub)
    conv_lb_map = Dict{Any,Any}(1 => conv_lb)

    ctx = ColGenContext(
        decomp, model,
        conv_ub_map, conv_lb_map,
        Dict{Any,Any}(),
        pool,
        NonRobustCutManager{CstrEq}(),
        Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}()
    )

    # Active branching constraint on x₁ = (1, 1)
    push!(ctx.branching_constraints,
        Vertigo.ColGen.ActiveBranchingConstraint(br, (1, 1))
    )

    return ctx, (
        c1=c1, c2=c2, conv_ub=conv_ub, conv_lb=conv_lb,
        br=br, z1=z1, z2=z2
    )
end

# ──────────────────────────────────────────────────────────────────

function test_insert_columns_single_column()
    @testset "[insert_columns] single column" begin
        ctx, refs = build_insert_columns_context()
        model = ctx.master_model

        # Column: z1 = 1.0
        # cost = 3.0, c1 = 2.0, c2 = 1.0, branching(x₁) = 1.0
        sol = SpSolution(1, 3.0, [(refs.z1, 1.0)])
        pricing_sol = Vertigo.ColGen.PricingPrimalSolution(
            1, sol, true
        )
        columns = Vertigo.ColGen.GeneratedColumns(Any[pricing_sol])

        n = Vertigo.ColGen.insert_columns!(ctx, Phase0(), columns)

        @test n == 1
        @test length(ctx.pool.by_master_var) == 1

        col_var = first(keys(ctx.pool.by_master_var))

        @test _obj_coeff(model, col_var) ≈ 3.0
        @test _constr_coeff(model, refs.c1, col_var) ≈ 2.0
        @test _constr_coeff(model, refs.c2, col_var) ≈ 1.0
        @test _constr_coeff(model, refs.conv_ub, col_var) ≈ 1.0
        @test _constr_coeff(model, refs.conv_lb, col_var) ≈ 1.0
        @test _constr_coeff(model, refs.br, col_var) ≈ 1.0
    end
end

function test_insert_columns_two_columns()
    @testset "[insert_columns] two columns" begin
        ctx, refs = build_insert_columns_context()
        model = ctx.master_model

        sol_a = SpSolution(1, 3.0, [(refs.z1, 1.0)])
        sol_b = SpSolution(1, 5.0, [(refs.z2, 1.0)])

        pricing_a = Vertigo.ColGen.PricingPrimalSolution(
            1, sol_a, true
        )
        pricing_b = Vertigo.ColGen.PricingPrimalSolution(
            1, sol_b, true
        )
        columns = Vertigo.ColGen.GeneratedColumns(
            Any[pricing_a, pricing_b]
        )

        n = Vertigo.ColGen.insert_columns!(ctx, Phase0(), columns)

        @test n == 2
        @test length(ctx.pool.by_master_var) == 2

        # Identify columns by cost
        col_vars = collect(keys(ctx.pool.by_master_var))
        costs = [_obj_coeff(model, v) for v in col_vars]
        var_a = col_vars[findfirst(c -> c ≈ 3.0, costs)]
        var_b = col_vars[findfirst(c -> c ≈ 5.0, costs)]

        # Column A (z1=1): c1=2, c2=1, br=1
        @test _constr_coeff(model, refs.c1, var_a) ≈ 2.0
        @test _constr_coeff(model, refs.c2, var_a) ≈ 1.0
        @test _constr_coeff(model, refs.br, var_a) ≈ 1.0

        # Column B (z2=1): c1=0, c2=4, br=0 (z2 maps to x₂, not x₁)
        @test _constr_coeff(model, refs.c1, var_b) ≈ 0.0
        @test _constr_coeff(model, refs.c2, var_b) ≈ 4.0
        @test _constr_coeff(model, refs.br, var_b) ≈ 0.0

        # Both have convexity = 1.0
        @test _constr_coeff(model, refs.conv_ub, var_a) ≈ 1.0
        @test _constr_coeff(model, refs.conv_ub, var_b) ≈ 1.0
        @test _constr_coeff(model, refs.conv_lb, var_a) ≈ 1.0
        @test _constr_coeff(model, refs.conv_lb, var_b) ≈ 1.0
    end
end

function test_insert_columns_duplicate_skipped()
    @testset "[insert_columns] duplicate column skipped" begin
        ctx, refs = build_insert_columns_context()

        sol = SpSolution(1, 3.0, [(refs.z1, 1.0)])
        pricing_sol = Vertigo.ColGen.PricingPrimalSolution(
            1, sol, true
        )

        n1 = Vertigo.ColGen.insert_columns!(
            ctx, Phase0(),
            Vertigo.ColGen.GeneratedColumns(Any[pricing_sol])
        )
        @test n1 == 1

        n2 = Vertigo.ColGen.insert_columns!(
            ctx, Phase0(),
            Vertigo.ColGen.GeneratedColumns(Any[pricing_sol])
        )
        @test n2 == 0
        @test length(ctx.pool.by_master_var) == 1
    end
end

function test_insert_columns_mixed_sp_variables()
    @testset "[insert_columns] column with two SP variables" begin
        ctx, refs = build_insert_columns_context()
        model = ctx.master_model

        # Column: z1=1, z2=2
        # cost = 3*1 + 5*2 = 13
        # c1 = 2*1 + 0*2 = 2, c2 = 1*1 + 4*2 = 9
        # br(x₁) = 1 (only z1 maps to x₁)
        sol = SpSolution(
            1, 13.0, [(refs.z1, 1.0), (refs.z2, 2.0)]
        )
        pricing_sol = Vertigo.ColGen.PricingPrimalSolution(
            1, sol, true
        )
        columns = Vertigo.ColGen.GeneratedColumns(Any[pricing_sol])

        n = Vertigo.ColGen.insert_columns!(ctx, Phase0(), columns)
        @test n == 1

        col_var = first(keys(ctx.pool.by_master_var))

        @test _obj_coeff(model, col_var) ≈ 13.0
        @test _constr_coeff(model, refs.c1, col_var) ≈ 2.0
        @test _constr_coeff(model, refs.c2, col_var) ≈ 9.0
        @test _constr_coeff(model, refs.conv_ub, col_var) ≈ 1.0
        @test _constr_coeff(model, refs.conv_lb, col_var) ≈ 1.0
        @test _constr_coeff(model, refs.br, col_var) ≈ 1.0
    end
end

function test_insert_columns_pool_records_original_cost()
    @testset "[insert_columns] pool records original cost" begin
        ctx, refs = build_insert_columns_context()

        sol = SpSolution(1, 3.0, [(refs.z1, 1.0)])
        pricing_sol = Vertigo.ColGen.PricingPrimalSolution(
            1, sol, true
        )
        columns = Vertigo.ColGen.GeneratedColumns(Any[pricing_sol])

        Vertigo.ColGen.insert_columns!(ctx, Phase0(), columns)

        col_var = first(keys(ctx.pool.by_master_var))
        entry = ctx.pool.by_master_var[col_var]

        @test entry.original_cost ≈ 3.0
        @test entry.sp_id == 1
    end
end

function test_insert_columns_empty_set()
    @testset "[insert_columns] empty column set" begin
        ctx, _ = build_insert_columns_context()

        columns = Vertigo.ColGen.GeneratedColumns(Any[])
        n = Vertigo.ColGen.insert_columns!(ctx, Phase0(), columns)

        @test n == 0
        @test isempty(ctx.pool.by_master_var)
    end
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

function test_insert_columns()
    test_insert_columns_single_column()
    test_insert_columns_two_columns()
    test_insert_columns_duplicate_skipped()
    test_insert_columns_mixed_sp_variables()
    test_insert_columns_pool_records_original_cost()
    test_insert_columns_empty_set()
end
