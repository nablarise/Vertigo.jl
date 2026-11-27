# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Vertigo.ColGen: get_master, _compute_sp_reduced_costs, _insert_column!,
    setup_stabilization!, check_misprice, column_reduced_cost,
    compute_dual_bound, PricingPrimalSolution, GeneratedColumns,
    MasterDualSolution, DualMoiSolution, Phase1, Phase2
using Vertigo.Reformulation: _SpSolution

# One subproblem used between 1 and 2 times, with a setup variable y of cost 1
# and two variables z of cost 1 in the coupling row `cover`.
# Duals: π_cover = 4.5, ν_lb = 1, ν_ub = -3.
# Dual objective: 4.5 × 1 + 1 × 1 - 3 × 2 = -0.5.
# Reduced costs, by hand: y: 1 - ν_lb - ν_ub = 3, z: 1 - π_cover = -3.5.
function _setup_variable_instance(; smoothing_alpha = 0.0)
    sp_id = PricingSubproblemId(1)

    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)
    @constraint(master_jump, cover, 0 >= 1)
    @constraint(master_jump, conv_lb, 0 >= 1)
    @constraint(master_jump, conv_ub, 0 <= 2)
    @objective(master_jump, Min, 0)

    sp_jump = Model(HiGHS.Optimizer)
    set_silent(sp_jump)
    @variable(sp_jump, y, Bin)
    @variable(sp_jump, z[1:2], Bin)
    @constraint(sp_jump, z[1] + z[2] <= 2 * y)

    builder = DWReformulationBuilder{Int}(minimize=true)
    add_subproblem!(builder, sp_id, 0.0, 1.0, 2.0)
    add_sp_variable!(builder, sp_id, index(y), 1.0)
    add_mapping!(builder, 0, sp_id, index(y))
    add_setup_variable!(builder, sp_id, index(y), index(conv_lb), index(conv_ub))
    for i in 1:2
        add_sp_variable!(builder, sp_id, index(z[i]), 1.0)
        add_coupling_coefficient!(builder, sp_id, index(z[i]), index(cover), 1.0)
        add_mapping!(builder, i, sp_id, index(z[i]))
    end
    add_coupling_constraint!(builder, index(cover), 1.0)
    decomp = build(builder)
    set_models!(
        decomp, backend(master_jump),
        Dict{PricingSubproblemId,Any}(sp_id => backend(sp_jump)),
        Dict(sp_id => TaggedCI(index(conv_ub))),
        Dict(sp_id => TaggedCI(index(conv_lb))),
    )

    ws = ColGenWorkspace(
        decomp, ColumnPool(),
        Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        ColGenConfig(smoothing_alpha=smoothing_alpha, silent=true),
    )
    duals = Dict(
        TaggedCI(index(cover)) => 4.5,
        TaggedCI(index(conv_lb)) => 1.0,
        TaggedCI(index(conv_ub)) => -3.0,
    )
    cc_ids = get_master(ws).coupling_constraint_ids
    dual_sol = MasterDualSolution(DualMoiSolution(-0.5, duals), cc_ids)
    rows = (cover = TaggedCI(index(cover)),
        conv_lb = TaggedCI(index(conv_lb)), conv_ub = TaggedCI(index(conv_ub)))
    return ws, sp_id, index(y), index.(z), rows, dual_sol
end

function _row_coefficient(model, row::TaggedCI, var)
    func = with_typed_ci(ci -> MOI.get(model, MOI.ConstraintFunction(), ci), row)
    return sum((t.coefficient for t in func.terms if t.variable == var); init = 0.0)
end

function test_setup_variable_reduced_costs()
    @testset "[setup_variable] the convexity rows are in the dual vector" begin
        ws, _, _, _, rows, dual_sol = _setup_variable_instance()
        @test rows.conv_lb in dual_sol.coupling_constraint_ids
        @test rows.conv_ub in dual_sol.coupling_constraint_ids
    end
    @testset "[setup_variable] its reduced cost pays the convexity duals" begin
        ws, sp_id, y, z, _, dual_sol = _setup_variable_instance()
        rc = _compute_sp_reduced_costs(ws, dual_sol, sp_id)
        # By hand: 1 - ν_lb - ν_ub = 1 - 1 + 3.
        @test rc[y] ≈ 3.0 atol = 1e-10
        # By hand: 1 - π_cover, with no convexity dual.
        @test rc[z[1]] ≈ -3.5 atol = 1e-10
    end
end

function test_setup_variable_column_reduced_cost()
    for phase in (Phase2(), Phase1())
        @testset "[setup_variable] column_reduced_cost adds nothing ($(nameof(typeof(phase))))" begin
            ws, sp_id, _, _, _, dual_sol = _setup_variable_instance()
            # Raw value of the column y = z1 = 1, by hand: 3 - 3.5.
            sol = _SpSolution(sp_id, -0.5, Tuple{MOI.VariableIndex,Float64}[])
            rc = column_reduced_cost(
                ws, phase, PricingPrimalSolution(sp_id, sol), dual_sol
            )
            @test rc ≈ -0.5 atol = 1e-10
        end
    end
end

function test_setup_variable_column_insertion()
    @testset "[setup_variable] a used copy has coefficient 1 in the convexity rows" begin
        ws, sp_id, y, z, rows, _ = _setup_variable_instance()
        sol = _SpSolution(sp_id, -0.5, [(y, 1.0), (z[1], 1.0)])
        col = _insert_column!(ws, Phase2(), PricingPrimalSolution(sp_id, sol))
        master = master_model(ws.decomp)
        @test _row_coefficient(master, rows.conv_lb, col) ≈ 1.0 atol = 1e-10
        @test _row_coefficient(master, rows.conv_ub, col) ≈ 1.0 atol = 1e-10
        @test _row_coefficient(master, rows.cover, col) ≈ 1.0 atol = 1e-10
    end
    @testset "[setup_variable] the empty column is not a copy" begin
        ws, sp_id, _, _, rows, _ = _setup_variable_instance()
        sol = _SpSolution(sp_id, 0.0, Tuple{MOI.VariableIndex,Float64}[])
        col = _insert_column!(ws, Phase2(), PricingPrimalSolution(sp_id, sol))
        master = master_model(ws.decomp)
        @test _row_coefficient(master, rows.conv_lb, col) ≈ 0.0 atol = 1e-10
        @test _row_coefficient(master, rows.conv_ub, col) ≈ 0.0 atol = 1e-10
    end
end

function test_setup_variable_dual_bound()
    @testset "[setup_variable] dual bound keeps the convexity term" begin
        ws, sp_id, _, _, _, dual_sol = _setup_variable_instance()
        # Subproblem optimum y = z1 = z2 = 1: 3 - 3.5 - 3.5 = -4.
        bound = compute_dual_bound(ws, Phase2(), Dict(sp_id => -4.0), dual_sol)
        # Lagrangian bound, by hand: π_cover × 1 + ν_lb × 1 + ν_ub × 2
        # + 2 copies × (-4) = 4.5 + 1 - 6 - 8.
        @test bound ≈ -8.5 atol = 1e-10
    end
end

function test_setup_variable_misprice()
    @testset "[setup_variable] check_misprice does not count ν twice" begin
        ws, sp_id, y, z, _, dual_sol = _setup_variable_instance(smoothing_alpha = 0.5)
        stab = setup_stabilization!(ws, nothing)
        # Reduced cost of y = z1 = 1 at the dual solution, by hand: 3 - 3.5.
        # It improves, so there is no misprice. Counting ν once more would
        # give -0.5 - 1 + 3 = 1.5, which does not improve.
        sol = _SpSolution(sp_id, -0.5, [(y, 1.0), (z[1], 1.0)])
        columns = GeneratedColumns([PricingPrimalSolution(sp_id, sol)])
        @test check_misprice(stab, columns, dual_sol) == false
    end
end

function test_setup_variable_builder()
    @testset "[setup_variable] a fixed cost next to a setup variable is rejected" begin
        sp_id = PricingSubproblemId(1)
        y = MOI.VariableIndex(1)
        conv_lb = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}(1)
        conv_ub = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}(2)
        builder = DWReformulationBuilder{Int}(minimize=true)
        add_subproblem!(builder, sp_id, 5.0, 0.0, 3.0)
        add_sp_variable!(builder, sp_id, y, 1.0)
        add_setup_variable!(builder, sp_id, y, conv_lb, conv_ub)
        @test_throws "setup variable and a fixed cost" build(builder)
    end
    @testset "[setup_variable] two setup variables are rejected" begin
        sp_id = PricingSubproblemId(1)
        conv_lb = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}(1)
        conv_ub = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}(2)
        builder = DWReformulationBuilder{Int}(minimize=true)
        add_subproblem!(builder, sp_id, 0.0, 0.0, 3.0)
        add_setup_variable!(builder, sp_id, MOI.VariableIndex(1), conv_lb, conv_ub)
        @test_throws "already has a setup variable" add_setup_variable!(
            builder, sp_id, MOI.VariableIndex(2), conv_lb, conv_ub
        )
    end
end

function test_setup_variable_e2e()
    test_setup_variable_builder()
    test_setup_variable_reduced_costs()
    test_setup_variable_column_reduced_cost()
    test_setup_variable_column_insertion()
    test_setup_variable_dual_bound()
    test_setup_variable_misprice()
end
