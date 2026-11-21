using Vertigo.ColGen: optimize_pricing_problem!, PricingSubproblem,
    SubproblemMoiOptimizer, MasterDualSolution, DualMoiSolution,
    get_primal_bound, get_dual_bound, get_primal_sols, compute_dual_bound,
    Phase2

# One subproblem with fixed cost 10, used between 1 and 2 times, whose
# objective -20 z1 - 5 z2 stands for the reduced costs of its variables.
# With z1 + z2 <= 1, its optimum is z1 = 1, of value -20.
function _pricing_bounds_instance()
    sp_id = PricingSubproblemId(1)

    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)
    @constraint(master_jump, cover, 0 >= 1)
    @constraint(master_jump, conv_lb, 0 >= 1)
    @constraint(master_jump, conv_ub, 0 <= 2)
    @objective(master_jump, Min, 0)

    sp_jump = Model(HiGHS.Optimizer)
    set_silent(sp_jump)
    @variable(sp_jump, z[1:2], Bin)
    @constraint(sp_jump, z[1] + z[2] <= 1)
    @objective(sp_jump, Min, -20 * z[1] - 5 * z[2])

    builder = DWReformulationBuilder{Int}(minimize=true)
    add_subproblem!(builder, sp_id, 10.0, 1.0, 2.0)
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
        ColGenConfig(silent=true),
    )
    # π_cover = 4, and both convexity duals nonzero: ν_lb = 1, ν_ub = -3.
    # Dual objective: 4 × 1 + 1 × 1 - 3 × 2 = -1.
    duals = Dict(
        TaggedCI(index(cover)) => 4.0,
        TaggedCI(index(conv_lb)) => 1.0,
        TaggedCI(index(conv_ub)) => -3.0,
    )
    dual_sol = MasterDualSolution(DualMoiSolution(-1.0, duals), TaggedCI[])
    return ws, sp_id, PricingSubproblem(backend(sp_jump)), dual_sol
end

function test_pricing_optimization_e2e()
    @testset "[pricing_optimization] MOI pricer bounds exclude fk and ν" begin
        ws, sp_id, pricing_sp, dual_sol = _pricing_bounds_instance()

        result = optimize_pricing_problem!(
            ws, sp_id, pricing_sp, SubproblemMoiOptimizer(), dual_sol, false
        )

        # Subproblem optimum, by hand: z1 = 1, value -20.
        @test get_primal_bound(result) ≈ -20.0 atol = 1e-6
        @test get_dual_bound(result) ≈ -20.0 atol = 1e-6
    end

    @testset "[pricing_optimization] MOI pricer column carries fk and ν" begin
        ws, sp_id, pricing_sp, dual_sol = _pricing_bounds_instance()

        result = optimize_pricing_problem!(
            ws, sp_id, pricing_sp, SubproblemMoiOptimizer(), dual_sol, false
        )

        # -20 + fk - ν_lb - ν_ub = -20 + 10 - 1 + 3 = -8, by hand.
        column = only(get_primal_sols(result))
        @test column.solution.obj_value ≈ -8.0 atol = 1e-6
    end

    @testset "[dual_bounds] bound from the raw subproblem value, fk and ν nonzero" begin
        ws, sp_id, _, dual_sol = _pricing_bounds_instance()

        bound = compute_dual_bound(ws, Phase2(), Dict(sp_id => -20.0), dual_sol)

        # Lagrangian bound, by hand: π_cover × 1 + min over λ in [1, 2] of
        # λ × (-20 + fk) = 4 + 2 × (-10) = -16.
        @test bound ≈ -16.0 atol = 1e-6
    end
end
