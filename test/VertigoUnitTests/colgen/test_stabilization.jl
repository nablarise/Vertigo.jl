# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Vertigo.ColGen: WentgesSmoothing, NoStabilization,
    setup_stabilization!, update_stabilization_after_master_optim!,
    get_stab_dual_sol, update_stabilization_after_pricing_optim!,
    check_misprice, update_stabilization_after_misprice!,
    update_stabilization_after_iter!, _convex_combination,
    MasterDualSolution, DualMoiSolution, GeneratedColumns,
    PricingPrimalSolution, _SpSolution, get_master,
    _get_convexity_dual, _dual_value, _compute_sp_reduced_costs,
    Phase0, Phase1, Phase2, TaggedCI,
    ActiveRobustCut, ActiveBranchingConstraint
using Vertigo.Reformulation: master_model, convexity_ub_ci, mapped_subproblem_var

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

function _make_dual_sol(
    obj::Float64,
    duals::Dict{TaggedCI,Float64}=Dict{TaggedCI,Float64}();
    cc_ids::Vector{TaggedCI}=TaggedCI[]
)
    return MasterDualSolution(
        DualMoiSolution(obj, duals), cc_ids
    )
end

"""Build a minimal ColGenWorkspace with smoothing_alpha for testing."""
function _build_stab_ctx(; alpha=0.5)
    inst = random_gap_instance(2, 4; seed=42)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    master_jump = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(master_jump)
    @constraint(master_jump, assignment[t in T], 0 == 1)
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)
    @objective(master_jump, Min, 0)
    master_model = JuMP.backend(master_jump)

    sp_models = Dict{PricingSubproblemId,Any}()
    sp_var_indices = Dict{PricingSubproblemId,Vector{MOI.VariableIndex}}()
    for k in K
        sp_jump = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(sp_jump)
        @variable(sp_jump, z[t in T], Bin)
        @constraint(sp_jump,
            sum(inst.weight[k, t] * z[t] for t in T) <= inst.capacity[k]
        )
        @objective(sp_jump, Min, sum(inst.cost[k, t] * z[t] for t in T))
        sp_models[PricingSubproblemId(k)] = JuMP.backend(sp_jump)
        sp_var_indices[PricingSubproblemId(k)] = [JuMP.index(z[t]) for t in T]
    end

    SpVar = MOI.VariableIndex
    CstrId = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    builder = DWReformulationBuilder{Tuple{Int,Int}}(
        minimize=true
    )
    for k in K
        add_subproblem!(builder, PricingSubproblemId(k), 0.0, 0.0, 1.0)
    end
    for k in K, t in T
        sp_var = sp_var_indices[PricingSubproblemId(k)][t]
        add_sp_variable!(builder, PricingSubproblemId(k), sp_var, inst.cost[k, t])
        cstr_idx = JuMP.index(assignment[t])
        add_coupling_coefficient!(builder, PricingSubproblemId(k), sp_var, cstr_idx, 1.0)
        add_mapping!(builder, (k, t), PricingSubproblemId(k), sp_var)
    end
    for t in T
        add_coupling_constraint!(
            builder, JuMP.index(assignment[t]), 1.0
        )
    end
    decomp = build(builder)

    pool = ColumnPool()
    conv_ub_map = Dict{PricingSubproblemId,TaggedCI}(
        PricingSubproblemId(k) => TaggedCI(JuMP.index(conv_ub[k])) for k in K
    )
    conv_lb_map = Dict{PricingSubproblemId,TaggedCI}(
        PricingSubproblemId(k) => TaggedCI(JuMP.index(conv_lb[k])) for k in K
    )

    set_models!(decomp, master_model, sp_models, conv_ub_map, conv_lb_map)

    config = ColGenConfig(smoothing_alpha=alpha)
    ws = ColGenWorkspace(decomp, pool,
        Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        config
    )
    return ws
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
        test_oracle_from_first_separation_point()
        test_oracle_reset_each_iteration()
        test_oracle_is_best_pricing_solution()
        test_oracle_keeps_non_improving_solution()
        test_oracle_skips_subproblem_without_solution()
    end
    @testset "[stabilization] misprice" begin
        test_misprice_inactive()
        test_misprice_deterministic_schedule()
        test_misprice_hard_cap()
    end
    @testset "[stabilization] iter_update" begin
        test_decrease_alpha()
        test_increase_alpha()
        test_angle_reads_robust_cut_row()
        test_angle_reads_branching_row()
        test_angle_ignores_convexity_row()
        test_angle_subtracts_column_activity_in_cut_row()
        test_angle_measured_from_iteration_in_point()
    end
end

function test_setup_construction()
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)
    @test stab isa WentgesSmoothing
    @test stab.smooth_dual_sol_coeff ≈ 0.5
    @test stab.cur_smooth_dual_sol_coeff ≈ 0.5
    @test isnothing(stab.stab_center)
    @test stab.best_lagrangian_bound == -Inf
    @test stab.nb_misprices == 0
end

function test_setup_no_smoothing()
    ws = _build_stab_ctx(alpha=0.0)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)
    @test stab isa NoStabilization
end

function test_convex_combination_arithmetic()
    EqCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    d_center = Dict{TaggedCI,Float64}(
        TaggedCI(EqCstr(1)) => 10.0, TaggedCI(EqCstr(2)) => 20.0
    )
    d_out = Dict{TaggedCI,Float64}(
        TaggedCI(EqCstr(1)) => 0.0, TaggedCI(EqCstr(2)) => 0.0
    )
    center = MasterDualSolution(DualMoiSolution(100.0, d_center), TaggedCI[])
    out = MasterDualSolution(DualMoiSolution(0.0, d_out), TaggedCI[])

    combined = _convex_combination(center, out, 0.5)
    @test combined.sol.constraint_duals[TaggedCI(EqCstr(1))] ≈ 5.0
    @test combined.sol.constraint_duals[TaggedCI(EqCstr(2))] ≈ 10.0
    @test combined.sol.obj_value ≈ 50.0
end

function test_convex_combination_disjoint_types()
    EqCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    LtCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}
    }
    d_center = Dict{TaggedCI,Float64}(
        TaggedCI(EqCstr(1)) => 4.0
    )
    d_out = Dict{TaggedCI,Float64}(
        TaggedCI(LtCstr(1)) => 6.0
    )
    center = MasterDualSolution(DualMoiSolution(10.0, d_center), TaggedCI[])
    out = MasterDualSolution(DualMoiSolution(20.0, d_out), TaggedCI[])

    combined = _convex_combination(center, out, 0.3)
    # EqCstr: 0.3*4.0 + 0.7*0.0 = 1.2
    @test combined.sol.constraint_duals[TaggedCI(EqCstr(1))] ≈ 1.2
    # LtCstr: 0.3*0.0 + 0.7*6.0 = 4.2
    @test combined.sol.constraint_duals[TaggedCI(LtCstr(1))] ≈ 4.2
    # obj: 0.3*10 + 0.7*20 = 17
    @test combined.sol.obj_value ≈ 17.0
end

function test_first_iteration_returns_false()
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)
    dual = _make_dual_sol(42.0)
    result = update_stabilization_after_master_optim!(stab, Phase0(), dual)
    @test result == false
    @test !isnothing(stab.stab_center)
    @test stab.stab_center.sol.obj_value ≈ 42.0
end

function test_subsequent_iteration_returns_true()
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)

    dual1 = _make_dual_sol(42.0)
    update_stabilization_after_master_optim!(stab, Phase0(), dual1)

    dual2 = _make_dual_sol(50.0)
    result = update_stabilization_after_master_optim!(stab, Phase0(), dual2)
    @test result == true
    @test stab.nb_misprices == 0
    @test stab.cur_smooth_dual_sol_coeff ≈ 0.5
end

function test_phase_change_resets_center()
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)

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
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)

    # Initialize the stab center
    dual1 = _make_dual_sol(0.0)
    update_stabilization_after_master_optim!(stab, Phase0(), dual1)

    sep_dual = _make_dual_sol(5.0)
    sp_sols = Dict{PricingSubproblemId,Vector{PricingPrimalSolution}}()

    # For minimization, improving means pseudo_db > best_lagrangian_bound
    update_stabilization_after_pricing_optim!(
        stab, ws, sp_sols, master, 10.0, sep_dual
    )
    @test stab.best_lagrangian_bound ≈ 10.0
    @test stab.stab_center.sol.obj_value ≈ 5.0
end

function test_no_improvement_keeps_center()
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)

    dual1 = _make_dual_sol(0.0)
    update_stabilization_after_master_optim!(stab, Phase0(), dual1)

    sep_dual = _make_dual_sol(5.0)
    sp_sols = Dict{PricingSubproblemId,Vector{PricingPrimalSolution}}()

    # First call sets bound to 10.0
    update_stabilization_after_pricing_optim!(
        stab, ws, sp_sols, master, 10.0, sep_dual
    )

    # Second call with worse bound (8.0 < 10.0) shouldn't update
    old_center = stab.stab_center
    sep_dual2 = _make_dual_sol(7.0)
    update_stabilization_after_pricing_optim!(
        stab, ws, sp_sols, master, 8.0, sep_dual2
    )
    @test stab.best_lagrangian_bound ≈ 10.0
    @test stab.stab_center === old_center
end

function test_misprice_inactive()
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)
    stab.cur_smooth_dual_sol_coeff = 0.0

    gen_cols = GeneratedColumns(PricingPrimalSolution[])
    dual = _make_dual_sol(0.0)
    @test check_misprice(stab, gen_cols, dual) == false
end

function test_misprice_deterministic_schedule()
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)
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

function test_misprice_hard_cap()
    # Use high alpha (0.95) so the formula alone would allow ~20
    # misprices, but the hard cap (default 10) should kick in first.
    ws = _build_stab_ctx(alpha=0.95)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)

    for _ in 1:9
        update_stabilization_after_misprice!(stab, nothing)
    end
    # After 9 misprices, formula gives α > 0
    @test stab.cur_smooth_dual_sol_coeff > 0.0

    # 10th misprice hits the cap → α forced to 0
    update_stabilization_after_misprice!(stab, nothing)
    @test stab.nb_misprices == 10
    @test stab.cur_smooth_dual_sol_coeff == 0.0
end

function test_decrease_alpha()
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)

    EqCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    d_center = Dict{TaggedCI,Float64}(
        TaggedCI(EqCstr(1)) => 1.0
    )
    d_out = Dict{TaggedCI,Float64}(
        TaggedCI(EqCstr(1)) => 3.0
    )
    center_dual = MasterDualSolution(DualMoiSolution(0.0, d_center), TaggedCI[])
    out_dual = MasterDualSolution(DualMoiSolution(0.0, d_out), TaggedCI[])

    stab.in_point = center_dual

    # Empty columns → g^sep_i = rhs_i (all 1.0 for GAP).
    # direction_product = 1.0 * (3.0 - 1.0) = 2.0 > 0 → decrease α
    stab.oracle_sols = Dict{PricingSubproblemId,_SpSolution}()

    old_alpha = stab.smooth_dual_sol_coeff
    update_stabilization_after_iter!(stab, out_dual)
    @test stab.smooth_dual_sol_coeff < old_alpha
    # α = max(0.0, 0.5 - 0.1) = 0.4
    @test stab.smooth_dual_sol_coeff ≈ 0.4 atol=1e-9
end

function test_increase_alpha()
    ws = _build_stab_ctx(alpha=0.5)
    master = get_master(ws)
    stab = setup_stabilization!(ws, master)

    EqCstr = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }
    d_center = Dict{TaggedCI,Float64}(
        TaggedCI(EqCstr(1)) => 3.0
    )
    d_out = Dict{TaggedCI,Float64}(
        TaggedCI(EqCstr(1)) => 1.0
    )
    center_dual = MasterDualSolution(DualMoiSolution(0.0, d_center), TaggedCI[])
    out_dual = MasterDualSolution(DualMoiSolution(0.0, d_out), TaggedCI[])

    stab.in_point = center_dual

    # Direction product = 1.0 * (1.0 - 3.0) = -2.0 ≤ 0 → increase α
    stab.oracle_sols = Dict{PricingSubproblemId,_SpSolution}()

    old_alpha = stab.smooth_dual_sol_coeff
    update_stabilization_after_iter!(stab, out_dual)
    @test stab.smooth_dual_sol_coeff > old_alpha
    # α = 0.5 + (1.0 - 0.5) * 0.1 = 0.55
    @test stab.smooth_dual_sol_coeff ≈ 0.55 atol=1e-9
end

# The angle test of the α update reads every dualized row, not only the
# coupling constraints (#117). In each case below, the coupling rows alone
# give a negative product (α would go up), and the other row makes it
# positive (α goes down), or the reverse for the activity case.

const _EqCstr = MOI.ConstraintIndex{
    MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
}

const _NO_ORACLE_SOLS = Dict{PricingSubproblemId,_SpSolution}()

# A master row with no column yet, `lhs >= rhs`.
function _add_empty_geq_row!(ws, rhs)
    f = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 0.0)
    return TaggedCI(MOI.add_constraint(master_model(ws.decomp), f, MOI.GreaterThan(rhs)))
end

# Stabilization whose center and LP duals differ on coupling row 2 by
# 3.0 → 1.0 (product 1 * (1 - 3) = -2, no column covers it) and on `extra`.
function _stab_for_angle(ws, extra_duals_center, extra_duals_out, oracle_sols)
    stab = setup_stabilization!(ws, get_master(ws))
    center = Dict{TaggedCI,Float64}(TaggedCI(_EqCstr(2)) => 3.0, extra_duals_center...)
    out = Dict{TaggedCI,Float64}(TaggedCI(_EqCstr(2)) => 1.0, extra_duals_out...)
    stab.in_point = MasterDualSolution(DualMoiSolution(0.0, center), TaggedCI[])
    stab.oracle_sols = oracle_sols
    return stab, MasterDualSolution(DualMoiSolution(0.0, out), TaggedCI[])
end

function test_angle_reads_robust_cut_row()
    ws = _build_stab_ctx(alpha=0.5)
    cut = _add_empty_geq_row!(ws, 2.0)
    push!(ws.robust_cuts, ActiveRobustCut(cut, Dict((1, 1) => 1.0)))
    # cut row: g = 2 (no column), dual 0 → 2: 2 * 2 = 4; total -2 + 4 = 2 > 0
    stab, out = _stab_for_angle(ws, [cut => 0.0], [cut => 2.0], _NO_ORACLE_SOLS)
    update_stabilization_after_iter!(stab, out)
    @test stab.smooth_dual_sol_coeff ≈ 0.4 atol=1e-9
end

function test_angle_reads_branching_row()
    ws = _build_stab_ctx(alpha=0.5)
    row = _add_empty_geq_row!(ws, 1.0)
    push!(ws.branching_constraints, ActiveBranchingConstraint(row, (1, 1)))
    # branching row: g = 1, dual 0 → 3: 1 * 3 = 3; total -2 + 3 = 1 > 0
    stab, out = _stab_for_angle(ws, [row => 0.0], [row => 3.0], _NO_ORACLE_SOLS)
    update_stabilization_after_iter!(stab, out)
    @test stab.smooth_dual_sol_coeff ≈ 0.4 atol=1e-9
end

# The convexity rows are not dualized, so their duals do not enter the
# Lagrangian function: a change of 0 → 3 on one of them would give +3 and
# flip the sign if it were read.
function test_angle_ignores_convexity_row()
    ws = _build_stab_ctx(alpha=0.5)
    conv_ub = convexity_ub_ci(ws.decomp, PricingSubproblemId(1))
    # coupling row 2 alone: -2 ≤ 0, so α goes up: 0.5 + 0.5 * 0.1 = 0.55
    stab, out = _stab_for_angle(ws, [conv_ub => 0.0], [conv_ub => 3.0], _NO_ORACLE_SOLS)
    update_stabilization_after_iter!(stab, out)
    @test stab.smooth_dual_sol_coeff ≈ 0.55 atol=1e-9
end

function test_angle_subtracts_column_activity_in_cut_row()
    ws = _build_stab_ctx(alpha=0.5)
    cut = _add_empty_geq_row!(ws, 2.0)
    push!(ws.robust_cuts, ActiveRobustCut(cut, Dict((1, 1) => 1.0)))
    # Subproblem 1's oracle solution is task 1, improving: multiplicity 1 (the
    # convexity upper bound). It covers coupling row 1 (no dual change) and
    # the cut: g = 2 - 1 = 1, dual 0 → 1.5: 1.5. Total -2 + 1.5 = -0.5 ≤ 0,
    # so α goes up. Without the activity, g = 2 gives +3 and α would go down.
    sp_id, sp_var = mapped_subproblem_var(ws.decomp, (1, 1))
    oracle_sols = Dict(sp_id => _SpSolution(sp_id, -5.0, [(sp_var, 1.0)]))
    stab, out = _stab_for_angle(ws, [cut => 0.0], [cut => 1.5], oracle_sols)
    update_stabilization_after_iter!(stab, out)
    @test stab.smooth_dual_sol_coeff ≈ 0.55 atol=1e-9
end

# A mispricing sequence ends with α̃ = 0, so the last separation point is the
# LP dual, and its improved bound makes it the new center. The angle is still
# measured from the center the iteration started from (Sadykov 2019, Step 6):
# from the new center, π^out - π^in would be 0 and α would go up.
function test_angle_measured_from_iteration_in_point()
    ws = _build_stab_ctx(alpha=0.5)
    stab = setup_stabilization!(ws, get_master(ws))
    center = _make_dual_sol(0.0, Dict{TaggedCI,Float64}(TaggedCI(_EqCstr(1)) => 1.0))
    out = _make_dual_sol(0.0, Dict{TaggedCI,Float64}(TaggedCI(_EqCstr(1)) => 3.0))
    update_stabilization_after_master_optim!(stab, Phase2(), center)
    update_stabilization_after_master_optim!(stab, Phase2(), out)
    update_stabilization_after_pricing_optim!(
        stab, ws, Dict{PricingSubproblemId,Vector{PricingPrimalSolution}}(),
        get_master(ws), 10.0, out
    )
    @test stab.stab_center === out
    # no column: g = rhs = 1 on row 1, 1 * (3 - 1) = 2 > 0: 0.5 - 0.1 = 0.4
    update_stabilization_after_iter!(stab, out)
    @test stab.smooth_dual_sol_coeff ≈ 0.4 atol=1e-9
end

# The angle test reads the subgradient at the first separation point of the
# iteration (Sadykov 2019, Step 6), from one solution per subproblem: the
# pricing optimum, improving or not.

function _sp1_solution(ws, obj_value)
    sp_id, sp_var = mapped_subproblem_var(ws.decomp, (1, 1))
    return _SpSolution(sp_id, obj_value, [(sp_var, 1.0)])
end

function _price!(stab, ws, sols; improving=true)
    sp_id = PricingSubproblemId(1)
    sp_sols = Dict(sp_id => [PricingPrimalSolution(sp_id, sol, improving) for sol in sols])
    update_stabilization_after_pricing_optim!(
        stab, ws, sp_sols, get_master(ws), -Inf, _make_dual_sol(0.0)
    )
end

function test_oracle_from_first_separation_point()
    ws = _build_stab_ctx(alpha=0.5)
    stab = setup_stabilization!(ws, get_master(ws))
    update_stabilization_after_master_optim!(stab, Phase2(), _make_dual_sol(0.0))
    first_sol = _sp1_solution(ws, -5.0)
    _price!(stab, ws, [first_sol])
    _price!(stab, ws, [_sp1_solution(ws, -3.0)])  # after a mispricing
    @test stab.oracle_sols[PricingSubproblemId(1)] === first_sol
end

function test_oracle_reset_each_iteration()
    ws = _build_stab_ctx(alpha=0.5)
    stab = setup_stabilization!(ws, get_master(ws))
    update_stabilization_after_master_optim!(stab, Phase2(), _make_dual_sol(0.0))
    _price!(stab, ws, [_sp1_solution(ws, -5.0)])
    update_stabilization_after_master_optim!(stab, Phase2(), _make_dual_sol(0.0))
    next_sol = _sp1_solution(ws, -3.0)
    _price!(stab, ws, [next_sol])
    @test stab.oracle_sols[PricingSubproblemId(1)] === next_sol
end

function test_oracle_is_best_pricing_solution()
    ws = _build_stab_ctx(alpha=0.5)
    stab = setup_stabilization!(ws, get_master(ws))
    update_stabilization_after_master_optim!(stab, Phase2(), _make_dual_sol(0.0))
    best = _sp1_solution(ws, -5.0)
    _price!(stab, ws, [_sp1_solution(ws, -1.0), best])
    @test stab.oracle_sols[PricingSubproblemId(1)] === best
end

function test_oracle_keeps_non_improving_solution()
    ws = _build_stab_ctx(alpha=0.5)
    stab = setup_stabilization!(ws, get_master(ws))
    update_stabilization_after_master_optim!(stab, Phase2(), _make_dual_sol(0.0))
    sol = _sp1_solution(ws, 3.0)
    _price!(stab, ws, [sol]; improving=false)
    @test stab.oracle_sols[PricingSubproblemId(1)] === sol
end

# A pricer that finds nothing below its threshold returns no solution.
function test_oracle_skips_subproblem_without_solution()
    ws = _build_stab_ctx(alpha=0.5)
    stab = setup_stabilization!(ws, get_master(ws))
    update_stabilization_after_master_optim!(stab, Phase2(), _make_dual_sol(0.0))
    _price!(stab, ws, _SpSolution[])
    @test isempty(stab.oracle_sols)
end
