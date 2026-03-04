# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.ColGen: WentgesSmoothing, NoStabilization,
    setup_stabilization!, update_stabilization_after_master_optim!,
    get_stab_dual_sol, update_stabilization_after_pricing_optim!,
    check_misprice, update_stabilization_after_misprice!,
    update_stabilization_after_iter!, _convex_combination,
    MasterDualSolution, DualMoiSolution, GeneratedColumns,
    PricingPrimalSolution, _SpSolution, get_master,
    _get_convexity_dual, _dual_value, _compute_sp_reduced_costs,
    Phase0, Phase1, Phase2, TaggedCI

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

function _make_dual_sol(
    obj::Float64, duals::Dict;
    cc_ids::Vector{TaggedCI}=TaggedCI[]
)
    constraint_duals = Dict{
        Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}
    }()
    for (ctype, inner) in duals
        constraint_duals[ctype] = inner
    end
    return MasterDualSolution(
        DualMoiSolution(obj, constraint_duals), cc_ids
    )
end

function _make_dual_sol(obj::Float64)
    return MasterDualSolution(
        DualMoiSolution(
            obj,
            Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}()
        ),
        TaggedCI[]
    )
end

"""Build a minimal ColGenContext with smoothing_alpha for testing."""
function _build_stab_ctx(; alpha=0.5)
    inst = random_gap_instance(2, 4)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    master_jump = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(master_jump)
    @constraint(master_jump, assignment[t in T], 0 == 1)
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)
    @objective(master_jump, Min, 0)
    master_model = JuMP.backend(master_jump)

    sp_models = Dict{Any,Any}()
    sp_var_indices = Dict{Int,Vector{MOI.VariableIndex}}()
    for k in K
        sp_jump = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(sp_jump)
        @variable(sp_jump, z[t in T], Bin)
        @constraint(sp_jump,
            sum(inst.weight[k, t] * z[t] for t in T) <= inst.capacity[k]
        )
        @objective(sp_jump, Min, sum(inst.cost[k, t] * z[t] for t in T))
        sp_models[k] = JuMP.backend(sp_jump)
        sp_var_indices[k] = [JuMP.index(z[t]) for t in T]
    end

    SpVar = MOI.VariableIndex
    CstrId = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    builder = DecompositionBuilder{Int,SpVar,Tuple{Int,Int},CstrId,Nothing}(
        minimize=true
    )
    for k in K
        add_subproblem!(builder, k, 0.0, 0.0, 1.0)
    end
    for k in K, t in T
        sp_var = sp_var_indices[k][t]
        add_sp_variable!(builder, k, sp_var, inst.cost[k, t])
        cstr_idx = JuMP.index(assignment[t])
        add_coupling_coefficient!(builder, k, sp_var, cstr_idx, 1.0)
        add_mapping!(builder, (k, t), k, sp_var)
    end
    for t in T
        add_coupling_constraint!(
            builder, JuMP.index(assignment[t]), EQUAL_TO, 1.0
        )
    end
    decomp = build(builder)

    pool = ColumnPool{MOI.VariableIndex,Int,SpVar}()
    conv_ub_map = Dict{Any,Any}(
        k => JuMP.index(conv_ub[k]) for k in K
    )
    conv_lb_map = Dict{Any,Any}(
        k => JuMP.index(conv_lb[k]) for k in K
    )

    ctx = ColGenContext(
        decomp, master_model, conv_ub_map, conv_lb_map, sp_models,
        pool, NonRobustCutManager{CstrId}(),
        Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}();
        smoothing_alpha=alpha
    )
    return ctx
end

# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────

function test_stabilization()
    @testset "[stabilization] setup" begin
        test_setup_construction()
        test_setup_no_smoothing()
    end
    @testset "[stabilization] convex_combination" begin
        test_convex_combination_arithmetic()
        test_convex_combination_disjoint_types()
    end
    @testset "[stabilization] after_master_optim" begin
        test_first_iteration_returns_false()
        test_subsequent_iteration_returns_true()
        test_phase_change_resets_center()
    end
    @testset "[stabilization] after_pricing_optim" begin
        test_bound_improvement_updates_center()
        test_no_improvement_keeps_center()
    end
    @testset "[stabilization] misprice" begin
        test_misprice_inactive()
        test_misprice_deterministic_schedule()
    end
    @testset "[stabilization] iter_update" begin
        test_decrease_alpha()
        test_increase_alpha()
    end
end

function test_setup_construction()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)
    @test stab isa WentgesSmoothing
    @test stab.smooth_dual_sol_coeff ≈ 0.5
    @test stab.cur_smooth_dual_sol_coeff ≈ 0.5
    @test isnothing(stab.stab_center)
    @test stab.best_lagrangian_bound == -Inf
    @test stab.nb_misprices == 0
end

function test_setup_no_smoothing()
    ctx = _build_stab_ctx(alpha=0.0)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)
    @test stab isa NoStabilization
end

function test_convex_combination_arithmetic()
    EqCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    d_center = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}(
        EqCstr => Dict(1 => 10.0, 2 => 20.0)
    )
    d_out = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}(
        EqCstr => Dict(1 => 0.0, 2 => 0.0)
    )
    center = MasterDualSolution(DualMoiSolution(100.0, d_center), TaggedCI[])
    out = MasterDualSolution(DualMoiSolution(0.0, d_out), TaggedCI[])

    combined = _convex_combination(center, out, 0.5)
    eq_duals = combined.sol.constraint_duals[EqCstr]
    @test eq_duals[1] ≈ 5.0
    @test eq_duals[2] ≈ 10.0
    @test combined.sol.obj_value ≈ 50.0
end

function test_convex_combination_disjoint_types()
    EqCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    LtCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}
    }
    d_center = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}(
        EqCstr => Dict(1 => 4.0)
    )
    d_out = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}(
        LtCstr => Dict(1 => 6.0)
    )
    center = MasterDualSolution(DualMoiSolution(10.0, d_center), TaggedCI[])
    out = MasterDualSolution(DualMoiSolution(20.0, d_out), TaggedCI[])

    combined = _convex_combination(center, out, 0.3)
    # EqCstr: 0.3*4.0 + 0.7*0.0 = 1.2
    @test combined.sol.constraint_duals[EqCstr][1] ≈ 1.2
    # LtCstr: 0.3*0.0 + 0.7*6.0 = 4.2
    @test combined.sol.constraint_duals[LtCstr][1] ≈ 4.2
    # obj: 0.3*10 + 0.7*20 = 17
    @test combined.sol.obj_value ≈ 17.0
end

function test_first_iteration_returns_false()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)
    dual = _make_dual_sol(42.0)
    result = update_stabilization_after_master_optim!(stab, Phase0(), dual)
    @test result == false
    @test !isnothing(stab.stab_center)
    @test stab.stab_center.sol.obj_value ≈ 42.0
end

function test_subsequent_iteration_returns_true()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)

    dual1 = _make_dual_sol(42.0)
    update_stabilization_after_master_optim!(stab, Phase0(), dual1)

    dual2 = _make_dual_sol(50.0)
    result = update_stabilization_after_master_optim!(stab, Phase0(), dual2)
    @test result == true
    @test stab.nb_misprices == 0
    @test stab.cur_smooth_dual_sol_coeff ≈ 0.5
end

function test_phase_change_resets_center()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)

    dual1 = _make_dual_sol(42.0)
    update_stabilization_after_master_optim!(stab, Phase0(), dual1)
    @test !isnothing(stab.stab_center)

    # Phase change should reset center
    dual2 = _make_dual_sol(10.0)
    result = update_stabilization_after_master_optim!(stab, Phase2(), dual2)
    @test result == false  # first iteration of new phase
    @test stab.stab_center.sol.obj_value ≈ 10.0
end

function test_bound_improvement_updates_center()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)

    # Initialize the stab center
    dual1 = _make_dual_sol(0.0)
    update_stabilization_after_master_optim!(stab, Phase0(), dual1)

    sep_dual = _make_dual_sol(5.0)
    gen_cols = GeneratedColumns(Any[])

    # For minimization, improving means pseudo_db > best_lagrangian_bound
    update_stabilization_after_pricing_optim!(
        stab, ctx, gen_cols, master, 10.0, sep_dual
    )
    @test stab.best_lagrangian_bound ≈ 10.0
    @test stab.stab_center.sol.obj_value ≈ 5.0
end

function test_no_improvement_keeps_center()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)

    dual1 = _make_dual_sol(0.0)
    update_stabilization_after_master_optim!(stab, Phase0(), dual1)

    sep_dual = _make_dual_sol(5.0)
    gen_cols = GeneratedColumns(Any[])

    # First call sets bound to 10.0
    update_stabilization_after_pricing_optim!(
        stab, ctx, gen_cols, master, 10.0, sep_dual
    )

    # Second call with worse bound (8.0 < 10.0) shouldn't update
    old_center = stab.stab_center
    sep_dual2 = _make_dual_sol(7.0)
    update_stabilization_after_pricing_optim!(
        stab, ctx, gen_cols, master, 8.0, sep_dual2
    )
    @test stab.best_lagrangian_bound ≈ 10.0
    @test stab.stab_center === old_center
end

function test_misprice_inactive()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)
    stab.cur_smooth_dual_sol_coeff = 0.0

    gen_cols = GeneratedColumns(Any[])
    dual = _make_dual_sol(0.0)
    @test check_misprice(stab, gen_cols, dual) == false
end

function test_misprice_deterministic_schedule()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)
    stab.smooth_dual_sol_coeff = 0.5

    # After 1 misprice: α = max(0, 1 - 1*(1-0.5)) = 0.5
    update_stabilization_after_misprice!(stab, nothing)
    @test stab.cur_smooth_dual_sol_coeff ≈ 0.5
    @test stab.nb_misprices == 1

    # After 2nd misprice: α = max(0, 1 - 2*(1-0.5)) = 0.0
    update_stabilization_after_misprice!(stab, nothing)
    @test stab.cur_smooth_dual_sol_coeff ≈ 0.0
    @test stab.nb_misprices == 2

    # After 3rd misprice: α = max(0, 1 - 3*0.5) = max(0, -0.5) = 0.0
    update_stabilization_after_misprice!(stab, nothing)
    @test stab.cur_smooth_dual_sol_coeff ≈ 0.0
    @test stab.nb_misprices == 3
end

function test_decrease_alpha()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)

    # Set up center
    EqCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    d_center = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}(
        EqCstr => Dict(1 => 1.0)
    )
    d_out = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}(
        EqCstr => Dict(1 => 3.0)
    )
    center_dual = MasterDualSolution(DualMoiSolution(0.0, d_center), TaggedCI[])
    out_dual = MasterDualSolution(DualMoiSolution(0.0, d_out), TaggedCI[])

    stab.stab_center = center_dual

    # Need generated columns for direction product computation.
    # Empty columns means g^sep_i = rhs_i (all 1.0 for our GAP).
    # Direction product = Σ rhs_i * (π^out_i - π^in_i)
    # For our GAP: 4 constraints, each rhs=1.0
    # All with (π^out - π^in) = 3.0 - 1.0 = 2.0 for constraint 1
    # and 0 - 0 = 0 for others
    # So direction_product = 1.0 * 2.0 = 2.0 > 0 → decrease α
    stab.last_generated_columns = GeneratedColumns(Any[])
    stab.last_sep_dual_sol = _make_dual_sol(0.0)

    old_alpha = stab.smooth_dual_sol_coeff
    update_stabilization_after_iter!(stab, out_dual)
    @test stab.smooth_dual_sol_coeff < old_alpha
end

function test_increase_alpha()
    ctx = _build_stab_ctx(alpha=0.5)
    master = get_master(ctx)
    stab = setup_stabilization!(ctx, master)

    EqCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    d_center = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}(
        EqCstr => Dict(1 => 3.0)
    )
    d_out = Dict{Type{<:MOI.ConstraintIndex},Dict{Int64,Float64}}(
        EqCstr => Dict(1 => 1.0)
    )
    center_dual = MasterDualSolution(DualMoiSolution(0.0, d_center), TaggedCI[])
    out_dual = MasterDualSolution(DualMoiSolution(0.0, d_out), TaggedCI[])

    stab.stab_center = center_dual

    # Direction product = 1.0 * (1.0 - 3.0) = -2.0 ≤ 0 → increase α
    stab.last_generated_columns = GeneratedColumns(Any[])
    stab.last_sep_dual_sol = _make_dual_sol(0.0)

    old_alpha = stab.smooth_dual_sol_coeff
    update_stabilization_after_iter!(stab, out_dual)
    @test stab.smooth_dual_sol_coeff > old_alpha
end
