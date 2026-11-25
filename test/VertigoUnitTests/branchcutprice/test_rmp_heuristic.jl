# One machine, one task: the pool and the pure master variables are filled
# by hand in each test, and the master objective constant is 0.
_one_task_gap_context() = build_gap_context(
    GAPInstance(1, 1, fill(1.0, 1, 1), fill(1.0, 1, 1), [1.0])
)

function test_rmp_heuristic()
    @testset "[rmp_heuristic] reported objective matching the columns passes" begin
        ws = _one_task_gap_context()
        mv1 = _add_test_column!(ws, 800, 1, 7.0)
        mv2 = _add_test_column!(ws, 801, 1, 3.0)
        var_values = Dict{MOI.VariableIndex,Float64}(mv1 => 1.0, mv2 => 2.0)

        # 7.0*1 + 3.0*2 = 13.0
        @test isnothing(
            _check_rmp_ip_objective(ws.decomp, ws.pool, var_values, 13.0)
        )
    end

    @testset "[rmp_heuristic] reported objective within relative tolerance passes" begin
        ws = _one_task_gap_context()
        mv = _add_test_column!(ws, 810, 1, 784.0)
        var_values = Dict{MOI.VariableIndex,Float64}(mv => 1.0)

        # |784.0001 - 784| = 1e-4 <= 1e-6 * 784
        @test isnothing(
            _check_rmp_ip_objective(ws.decomp, ws.pool, var_values, 784.0001)
        )
    end

    @testset "[rmp_heuristic] reported objective off the columns throws" begin
        # The HiGHS failure of #106: the solution costs 784, the solver
        # reports 326.
        ws = _one_task_gap_context()
        mv1 = _add_test_column!(ws, 820, 1, 400.0)
        mv2 = _add_test_column!(ws, 821, 1, 384.0)
        mv3 = _add_test_column!(ws, 822, 1, 50.0)
        var_values = Dict{MOI.VariableIndex,Float64}(
            mv1 => 1.0, mv2 => 1.0, mv3 => 0.0
        )

        @test_throws ["reported 326.0", "costs 784.0"] (
            _check_rmp_ip_objective(ws.decomp, ws.pool, var_values, 326.0)
        )
    end

    @testset "[rmp_heuristic] pure master variables are costed" begin
        ws = _one_task_gap_context()
        mv = _add_test_column!(ws, 830, 1, 5.0)
        pmv_int = _add_test_pure_master_var!(ws, 831, 3.0, true)
        pmv_cont = _add_test_pure_master_var!(ws, 832, 4.0, false)
        var_values = Dict{MOI.VariableIndex,Float64}(
            mv => 1.0, pmv_int => 2.0, pmv_cont => 0.75
        )

        # 5.0*1 + 3.0*2 + 4.0*0.75 = 14.0
        @test isnothing(
            _check_rmp_ip_objective(ws.decomp, ws.pool, var_values, 14.0)
        )
        # 5.0 counts the column alone.
        @test_throws ["reported 5.0", "costs 14.0"] (
            _check_rmp_ip_objective(ws.decomp, ws.pool, var_values, 5.0)
        )
    end

    @testset "[rmp_heuristic] master objective constant is counted" begin
        ws = _one_task_gap_context()
        MOI.modify(
            master_model(ws.decomp),
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarConstantChange(100.0)
        )
        mv = _add_test_column!(ws, 840, 1, 7.0)
        var_values = Dict{MOI.VariableIndex,Float64}(mv => 1.0)

        # 100 + 7.0*1 = 107.0
        @test isnothing(
            _check_rmp_ip_objective(ws.decomp, ws.pool, var_values, 107.0)
        )
        @test_throws ["reported 7.0", "costs 107.0"] (
            _check_rmp_ip_objective(ws.decomp, ws.pool, var_values, 7.0)
        )
    end
end
