# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

function test_dml_identical_index_set()
    @testset "[identical] a block carrying the multiplicities" begin
        K = identical(:bins, 4; lm = 1)
        @test K.name == :bins
        @test K.lm == 1
        @test K.um == 4
    end
    @testset "[identical] an index set whose one element is itself" begin
        K = identical(:bins, 4)
        @test length(K) == 1
        @test collect(K) == [K]
    end
    @testset "[identical] the block indexes a JuMP container as a variable" begin
        K = identical(:bins, 4)
        model = Model()
        @variable(model, x[k in K, i in 1:2])
        @test x[K, 1] isa VariableRef
        @test [x[k, 1] for k in K] == [x[K, 1]]
    end
    @testset "[identical] lm defaults to 0" begin
        @test identical(:bins, 4).lm == 0
    end
    @testset "[identical] sets with the same name and multiplicities are equal" begin
        @test identical(:bins, 4) == identical(:bins, 4)
        @test hash(identical(:bins, 4)) == hash(identical(:bins, 4))
    end
    @testset "[identical] sets with different names are different" begin
        @test identical(:small, 4) != identical(:large, 4)
    end
    @testset "[identical] sorted by name" begin
        a = identical(:a, 9)
        b = identical(:b, 1)
        @test sort([b, a]) == [a, b]
    end
    for (case, um, lm) in [
        ("lm > um", 2, 3),
        ("lm < 0", 2, -1),
        ("infinite um", Inf, 0),
        ("fractional um", 2.5, 0),
        ("fractional lm", 2, 0.5),
    ]
        @testset "[identical] rejects $case" begin
            @test_throws ArgumentError identical(:bins, um; lm = lm)
        end
    end
end

function test_dml_identical_multiplicity()
    @testset "[identical] _multiplicity" begin
        @test Vertigo.DML._multiplicity(identical(:bins, 3; lm = 1)) == (1.0, 3.0)
        @test Vertigo.DML._multiplicity(2) == (1.0, 1.0)
        @test Vertigo.DML._multiplicity((:machine, "a")) == (1.0, 1.0)
    end
end

_convexity_set(decomp, ci) =
    with_typed_ci(c -> MOI.get(master_model(decomp), MOI.ConstraintSet(), c), ci)

function test_dml_identical_decomposition_bounds()
    big = identical(:big, 3; lm = 1)
    small = identical(:small, 2)
    K = [big, small]
    model = Model()
    @variable(model, x[k in K, i in 1:2], Bin)
    @constraint(model, cover[i in 1:2], sum(x[k, i] for k in K) >= 1)
    @objective(model, Min, sum(x))
    decomp, sp_map = @dantzig_wolfe model begin
        x[k, _] => subproblem(k)
        cover[_] => master()
    end

    for (block, lm, um) in [(big, 1.0, 3.0), (small, 0.0, 2.0)]
        sp_id = sp_map[block]
        @testset "[identical] $block: bounds of the decomposition" begin
            @test convexity_bounds(decomp, sp_id) == (lm, um)
        end
        @testset "[identical] $block: bounds of the master rows" begin
            @test _convexity_set(decomp, convexity_lb_ci(decomp, sp_id)) == MOI.GreaterThan(lm)
            @test _convexity_set(decomp, convexity_ub_ci(decomp, sp_id)) == MOI.LessThan(um)
        end
    end
    @testset "[identical] a subproblem of another id has one copy" begin
        decomp, sp_map = @dantzig_wolfe _build_gap_jump_model() begin
            x[k, _] => subproblem(k)
            assign[_] => master()
            knapsack[k] => subproblem(k)
        end
        sp_id = sp_map[1]
        @test convexity_bounds(decomp, sp_id) == (1.0, 1.0)
        @test _convexity_set(decomp, convexity_lb_ci(decomp, sp_id)) == MOI.GreaterThan(1.0)
        @test _convexity_set(decomp, convexity_ub_ci(decomp, sp_id)) == MOI.LessThan(1.0)
    end
end

function test_dml_single_variable_objective()
    @testset "[dml] a single-variable objective becomes affine" begin
        K = identical(:bins, 2)
        model = Model()
        @variable(model, x[k in K, i in 1:2], Bin)
        @variable(model, y[k in K], Bin)
        @constraint(model, knapsack[k in K], x[k, 1] + x[k, 2] <= y[k])
        @constraint(model, cover[i in 1:2], sum(x[k, i] for k in K) >= 1)
        @objective(model, Min, sum(y[k] for k in K))
        @assert objective_function(model) isa VariableRef

        decomp, sp_map = @dantzig_wolfe model begin
            x[k, _] => subproblem(k)
            y[k] => subproblem(k)
            knapsack[k] => subproblem(k)
            cover[_] => master()
        end
        saf = MOI.ScalarAffineFunction{Float64}
        @test MOI.get(master_model(decomp), MOI.ObjectiveFunctionType()) == saf
        @test MOI.get(sp_model(decomp, sp_map[K]), MOI.ObjectiveFunctionType()) == saf
    end
end

function test_identical()
    test_dml_identical_index_set()
    test_dml_identical_multiplicity()
    test_dml_identical_decomposition_bounds()
    test_dml_single_variable_objective()
end
