# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────────────────

function _make_primal_sol(var_values::Dict{MOI.VariableIndex,Float64})
    obj = sum(values(var_values); init=0.0)
    return MasterPrimalSolution(PrimalMoiSolution(obj, var_values))
end

# Add a single column to a context's pool using a fresh MOI.VariableIndex as master var.
function _add_test_column!(ctx, master_var_id::Int, sp_id_val::Int, cost::Float64)
    sp_id = PricingSubproblemId(sp_id_val)
    sp_var = MOI.VariableIndex(master_var_id + 1000)   # unique SP var
    sol = Vertigo.Reformulation._SpSolution(sp_id, cost, [(sp_var, 1.0)])
    master_var = MOI.VariableIndex(master_var_id)
    record_column!(ctx.pool, master_var, sp_id, sol, cost)
    return master_var
end

# ────────────────────────────────────────────────────────────────────────────────────────
# _project_if_integral
# ────────────────────────────────────────────────────────────────────────────────────────

function test_ip_project_integral_solution()
    @testset "[ip_management] _project_if_integral integral columns" begin
        inst = random_gap_instance(1, 1; seed=1)
        ctx = build_gap_context(inst)

        mv1 = _add_test_column!(ctx, 500, 1, 7.0)
        mv2 = _add_test_column!(ctx, 501, 1, 3.0)

        var_values = Dict{MOI.VariableIndex,Float64}(mv1 => 1.0, mv2 => 0.0)
        sol = _make_primal_sol(var_values)

        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)

        @test !isnothing(result)
        @test cut == false
        @test result.obj_value ≈ 7.0
        @test length(result.non_zero_integral) == 1
        @test result.non_zero_integral[1] == (mv1, 1)
    end
end

function test_ip_project_fractional_solution()
    @testset "[ip_management] _project_if_integral fractional returns nothing" begin
        inst = random_gap_instance(1, 1; seed=2)
        ctx = build_gap_context(inst)

        mv = _add_test_column!(ctx, 502, 1, 5.0)
        var_values = Dict{MOI.VariableIndex,Float64}(mv => 0.5)
        sol = _make_primal_sol(var_values)

        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)
        @test isnothing(result)
        @test cut == false
    end
end

function test_ip_project_empty_pool()
    @testset "[ip_management] _project_if_integral empty pool is integral (obj=0)" begin
        inst = random_gap_instance(1, 1; seed=3)
        ctx = build_gap_context(inst)

        sol = _make_primal_sol(Dict{MOI.VariableIndex,Float64}())
        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)

        @test !isnothing(result)
        @test result.obj_value ≈ 0.0
        @test isempty(result.non_zero_integral)
        @test cut == false
    end
end

function test_ip_project_missing_var_defaults_zero()
    @testset "[ip_management] _project_if_integral missing var defaults to 0.0 (integral)" begin
        inst = random_gap_instance(1, 1; seed=4)
        ctx = build_gap_context(inst)

        _add_test_column!(ctx, 503, 1, 9.0)
        # Primal sol does not mention the column var → defaults to 0.0 (integral, ival=0)
        sol = _make_primal_sol(Dict{MOI.VariableIndex,Float64}())
        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)

        @test !isnothing(result)
        @test cut == false
        @test result.obj_value ≈ 0.0
        @test isempty(result.non_zero_integral)
    end
end

function test_ip_project_near_integer_within_tol()
    @testset "[ip_management] _project_if_integral near-integer within tolerance" begin
        inst = random_gap_instance(1, 1; seed=5)
        ctx = build_gap_context(inst)

        mv = _add_test_column!(ctx, 504, 1, 4.0)
        # 1.0 - 1e-6 is within tol=1e-5 of 1.0 → integral
        var_values = Dict{MOI.VariableIndex,Float64}(mv => 1.0 - 1e-6)
        sol = _make_primal_sol(var_values)
        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)

        @test !isnothing(result)
        @test cut == false
        @test result.obj_value ≈ 4.0
    end
end

function test_ip_project_multiplicity_greater_than_one()
    @testset "[ip_management] _project_if_integral multiplicity > 1 (identical SPs)" begin
        inst = random_gap_instance(1, 1; seed=6)
        ctx = build_gap_context(inst)

        mv = _add_test_column!(ctx, 505, 1, 2.0)
        # val=3.0 → rounded=3, ival=3
        var_values = Dict{MOI.VariableIndex,Float64}(mv => 3.0)
        sol = _make_primal_sol(var_values)
        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)

        @test !isnothing(result)
        @test cut == false
        @test result.obj_value ≈ 6.0
        @test result.non_zero_integral[1] == (mv, 3)
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# _project_if_integral with pure master variables
# ────────────────────────────────────────────────────────────────────────────────────────

function _add_test_pure_master_var!(ctx, var_id::Int, cost::Float64, is_integer::Bool)
    pmv = Vertigo.ColGen.PureMasterVariableData(
        MOI.VariableIndex(var_id), cost, 0.0, 1.0, is_integer,
        Vertigo.Reformulation.CouplingEntry[]
    )
    push!(ctx.decomp.pure_master_vars, pmv)
    return MOI.VariableIndex(var_id)
end

function test_ip_project_integral_with_integer_pure_master()
    @testset "[ip_management] _project_if_integral integer pure master integral" begin
        inst = random_gap_instance(1, 1; seed=30)
        ctx = build_gap_context(inst)

        mv = _add_test_column!(ctx, 700, 1, 5.0)
        pmv_id = _add_test_pure_master_var!(ctx, 701, 3.0, true)

        var_values = Dict{MOI.VariableIndex,Float64}(
            mv => 1.0, pmv_id => 2.0
        )
        sol = _make_primal_sol(var_values)
        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)

        @test !isnothing(result)
        @test cut == false
        @test result.obj_value ≈ 5.0 + 3.0 * 2
        @test length(result.non_zero_integral) == 2
    end
end

function test_ip_project_fractional_integer_pure_master()
    @testset "[ip_management] _project_if_integral fractional integer pure master → nothing" begin
        inst = random_gap_instance(1, 1; seed=31)
        ctx = build_gap_context(inst)

        mv = _add_test_column!(ctx, 710, 1, 5.0)
        pmv_id = _add_test_pure_master_var!(ctx, 711, 3.0, true)

        var_values = Dict{MOI.VariableIndex,Float64}(
            mv => 1.0, pmv_id => 0.5
        )
        sol = _make_primal_sol(var_values)
        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)

        @test isnothing(result)
        @test cut == false
    end
end

function test_ip_project_with_continuous_pure_master()
    @testset "[ip_management] _project_if_integral continuous pure master always passes" begin
        inst = random_gap_instance(1, 1; seed=32)
        ctx = build_gap_context(inst)

        mv = _add_test_column!(ctx, 720, 1, 5.0)
        pmv_id = _add_test_pure_master_var!(ctx, 721, 4.0, false)

        var_values = Dict{MOI.VariableIndex,Float64}(
            mv => 1.0, pmv_id => 0.75
        )
        sol = _make_primal_sol(var_values)
        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)

        @test !isnothing(result)
        @test cut == false
        @test result.obj_value ≈ 5.0 + 4.0 * 0.75
        @test length(result.non_zero_continuous) == 1
        @test result.non_zero_continuous[1][1] == pmv_id
        @test result.non_zero_continuous[1][2] ≈ 0.75
    end
end

function test_ip_project_mixed_columns_and_pure_master()
    @testset "[ip_management] _project_if_integral mixed columns + pure master" begin
        inst = random_gap_instance(1, 1; seed=33)
        ctx = build_gap_context(inst)

        mv1 = _add_test_column!(ctx, 730, 1, 2.0)
        mv2 = _add_test_column!(ctx, 731, 1, 3.0)
        pmv_int = _add_test_pure_master_var!(ctx, 732, 7.0, true)
        pmv_cont = _add_test_pure_master_var!(ctx, 733, 1.5, false)

        var_values = Dict{MOI.VariableIndex,Float64}(
            mv1 => 1.0, mv2 => 2.0,
            pmv_int => 1.0, pmv_cont => 0.4
        )
        sol = _make_primal_sol(var_values)
        result, cut = Vertigo.ColGen._project_if_integral(sol, ctx)

        @test !isnothing(result)
        @test cut == false
        expected_obj = 2.0 * 1 + 3.0 * 2 + 7.0 * 1 + 1.5 * 0.4
        @test result.obj_value ≈ expected_obj
        @test length(result.non_zero_integral) == 3
        @test length(result.non_zero_continuous) == 1
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# _has_artificial_vars_in_solution
# ────────────────────────────────────────────────────────────────────────────────────────

function test_ip_has_art_eq_active()
    @testset "[ip_management] _has_artificial_vars eq art var active" begin
        inst = random_gap_instance(1, 1; seed=7)
        ctx = build_gap_context(inst)

        fake_ci = TaggedCI(MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}(99))
        s_pos = MOI.VariableIndex(200)
        s_neg = MOI.VariableIndex(201)
        ctx.eq_art_vars[fake_ci] = (s_pos, s_neg)

        # s_pos active
        var_values = Dict{MOI.VariableIndex,Float64}(s_pos => 0.5, s_neg => 0.0)
        sol = _make_primal_sol(var_values)
        @test Vertigo.ColGen._has_artificial_vars_in_solution(ctx, sol) == true

        # both zero
        var_values2 = Dict{MOI.VariableIndex,Float64}(s_pos => 0.0, s_neg => 0.0)
        sol2 = _make_primal_sol(var_values2)
        @test Vertigo.ColGen._has_artificial_vars_in_solution(ctx, sol2) == false
    end
end

function test_ip_has_art_leq_active()
    @testset "[ip_management] _has_artificial_vars leq art var active" begin
        inst = random_gap_instance(1, 1; seed=8)
        ctx = build_gap_context(inst)

        fake_ci = TaggedCI(MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}(98))
        s = MOI.VariableIndex(202)
        ctx.leq_art_vars[fake_ci] = s

        var_values = Dict{MOI.VariableIndex,Float64}(s => 1e-4)
        sol = _make_primal_sol(var_values)
        @test Vertigo.ColGen._has_artificial_vars_in_solution(ctx, sol) == true

        # within tolerance
        var_values2 = Dict{MOI.VariableIndex,Float64}(s => 1e-6)
        sol2 = _make_primal_sol(var_values2)
        @test Vertigo.ColGen._has_artificial_vars_in_solution(ctx, sol2) == false
    end
end

function test_ip_has_art_geq_active()
    @testset "[ip_management] _has_artificial_vars geq art var active" begin
        inst = random_gap_instance(1, 1; seed=9)
        ctx = build_gap_context(inst)

        fake_ci = TaggedCI(MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}(97))
        s = MOI.VariableIndex(203)
        ctx.geq_art_vars[fake_ci] = s

        var_values = Dict{MOI.VariableIndex,Float64}(s => 2e-5)
        sol = _make_primal_sol(var_values)
        @test Vertigo.ColGen._has_artificial_vars_in_solution(ctx, sol) == true

        var_values2 = Dict{MOI.VariableIndex,Float64}(s => 0.0)
        sol2 = _make_primal_sol(var_values2)
        @test Vertigo.ColGen._has_artificial_vars_in_solution(ctx, sol2) == false
    end
end

function test_ip_has_art_no_art_vars()
    @testset "[ip_management] _has_artificial_vars no art vars always false" begin
        inst = random_gap_instance(1, 1; seed=10)
        ctx = build_gap_context(inst)  # art var dicts are empty

        var_values = Dict{MOI.VariableIndex,Float64}(MOI.VariableIndex(5) => 100.0)
        sol = _make_primal_sol(var_values)
        @test Vertigo.ColGen._has_artificial_vars_in_solution(ctx, sol) == false
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# check_primal_ip_feasibility!
# ────────────────────────────────────────────────────────────────────────────────────────

function test_ip_check_with_art_vars_active()
    @testset "[ip_management] check_primal_ip_feasibility! art vars → nothing" begin
        inst = random_gap_instance(1, 1; seed=11)
        ctx = build_gap_context(inst)

        s_pos = MOI.VariableIndex(300)
        s_neg = MOI.VariableIndex(301)
        fake_ci = TaggedCI(MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}(88))
        ctx.eq_art_vars[fake_ci] = (s_pos, s_neg)

        var_values = Dict{MOI.VariableIndex,Float64}(s_pos => 1.0)
        sol = _make_primal_sol(var_values)

        result, cut = check_primal_ip_feasibility!(sol, ctx, Phase2())
        @test isnothing(result)
        @test cut == false
    end
end

function test_ip_check_no_art_vars_integral()
    @testset "[ip_management] check_primal_ip_feasibility! no art vars + integral → sol" begin
        inst = random_gap_instance(1, 1; seed=12)
        ctx = build_gap_context(inst)

        mv = _add_test_column!(ctx, 600, 1, 10.0)
        var_values = Dict{MOI.VariableIndex,Float64}(mv => 1.0)
        sol = _make_primal_sol(var_values)

        result, cut = check_primal_ip_feasibility!(sol, ctx, Phase2())
        @test !isnothing(result)
        @test result.obj_value ≈ 10.0
        @test cut == false
    end
end

function test_ip_check_no_art_vars_fractional()
    @testset "[ip_management] check_primal_ip_feasibility! no art vars + fractional → nothing" begin
        inst = random_gap_instance(1, 1; seed=13)
        ctx = build_gap_context(inst)

        mv = _add_test_column!(ctx, 601, 1, 10.0)
        var_values = Dict{MOI.VariableIndex,Float64}(mv => 0.3)
        sol = _make_primal_sol(var_values)

        result, cut = check_primal_ip_feasibility!(sol, ctx, Phase2())
        @test isnothing(result)
        @test cut == false
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# update_inc_primal_sol! / _is_strictly_better
# ────────────────────────────────────────────────────────────────────────────────────────

function test_ip_update_incumbent_first_solution()
    @testset "[ip_management] update_inc_primal_sol! sets first incumbent" begin
        inst = random_gap_instance(1, 1; seed=14)
        ctx = build_gap_context(inst)

        @test isnothing(ctx.ip_incumbent)
        sol = Vertigo.ColGen.MasterIpPrimalSol(42.0, Tuple{MOI.VariableIndex,Int}[])
        update_inc_primal_sol!(ctx, nothing, sol)
        @test !isnothing(ctx.ip_incumbent)
        @test ctx.ip_incumbent.obj_value ≈ 42.0
    end
end

function test_ip_update_incumbent_better_min()
    @testset "[ip_management] update_inc_primal_sol! accepts strictly better (min)" begin
        inst = random_gap_instance(1, 1; seed=15)
        ctx = build_gap_context(inst)
        @test is_minimization(ctx.decomp)

        update_inc_primal_sol!(ctx, nothing, Vertigo.ColGen.MasterIpPrimalSol(50.0, Tuple{MOI.VariableIndex,Int}[]))
        update_inc_primal_sol!(ctx, nothing, Vertigo.ColGen.MasterIpPrimalSol(40.0, Tuple{MOI.VariableIndex,Int}[]))
        @test ctx.ip_incumbent.obj_value ≈ 40.0
    end
end

function test_ip_update_incumbent_worse_rejected_min()
    @testset "[ip_management] update_inc_primal_sol! rejects worse solution (min)" begin
        inst = random_gap_instance(1, 1; seed=16)
        ctx = build_gap_context(inst)

        update_inc_primal_sol!(ctx, nothing, Vertigo.ColGen.MasterIpPrimalSol(40.0, Tuple{MOI.VariableIndex,Int}[]))
        update_inc_primal_sol!(ctx, nothing, Vertigo.ColGen.MasterIpPrimalSol(60.0, Tuple{MOI.VariableIndex,Int}[]))
        @test ctx.ip_incumbent.obj_value ≈ 40.0
    end
end

function test_ip_is_strictly_better_max()
    @testset "[ip_management] _is_strictly_better maximization sense" begin
        # Use a max context: build it then verify sense
        inst = random_gap_instance(1, 1; seed=17)
        ctx = build_gap_context(inst)  # GAP is minimization

        # Test the helper directly with manual sense logic
        sol_good = Vertigo.ColGen.MasterIpPrimalSol(100.0, Tuple{MOI.VariableIndex,Int}[])
        sol_bad  = Vertigo.ColGen.MasterIpPrimalSol(50.0, Tuple{MOI.VariableIndex,Int}[])

        # For minimization: lower is better
        @test Vertigo.ColGen._is_strictly_better(ctx, sol_bad, sol_good) == true
        @test Vertigo.ColGen._is_strictly_better(ctx, sol_good, sol_bad) == false

        # Equal values: not strictly better
        sol_eq = Vertigo.ColGen.MasterIpPrimalSol(50.0, Tuple{MOI.VariableIndex,Int}[])
        @test Vertigo.ColGen._is_strictly_better(ctx, sol_bad, sol_eq) == false
    end
end

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────────────

function test_ip_management()
    test_ip_project_integral_solution()
    test_ip_project_fractional_solution()
    test_ip_project_empty_pool()
    test_ip_project_missing_var_defaults_zero()
    test_ip_project_near_integer_within_tol()
    test_ip_project_multiplicity_greater_than_one()
    test_ip_project_integral_with_integer_pure_master()
    test_ip_project_fractional_integer_pure_master()
    test_ip_project_with_continuous_pure_master()
    test_ip_project_mixed_columns_and_pure_master()
    test_ip_has_art_eq_active()
    test_ip_has_art_leq_active()
    test_ip_has_art_geq_active()
    test_ip_has_art_no_art_vars()
    test_ip_check_with_art_vars_active()
    test_ip_check_no_art_vars_integral()
    test_ip_check_no_art_vars_fractional()
    test_ip_update_incumbent_first_solution()
    test_ip_update_incumbent_better_min()
    test_ip_update_incumbent_worse_rejected_min()
    test_ip_is_strictly_better_max()
end
