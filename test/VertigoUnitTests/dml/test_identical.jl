# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

function test_dml_identical_index_set()
    @testset "[identical] one element carrying the bounds" begin
        K = identical(:bins, 4; lb = 1)
        @test length(K) == 1
        @test only(K).name == :bins
        @test only(K).lb == 1
        @test only(K).ub == 4
    end
    @testset "[identical] lb defaults to 0" begin
        @test only(identical(:bins, 4)).lb == 0
    end
    @testset "[identical] sets with the same name and bounds are equal" begin
        @test only(identical(:bins, 4)) == only(identical(:bins, 4))
        @test hash(only(identical(:bins, 4))) == hash(only(identical(:bins, 4)))
    end
    @testset "[identical] sets with different names are different" begin
        @test only(identical(:small, 4)) != only(identical(:large, 4))
    end
    @testset "[identical] sorted by name" begin
        a = only(identical(:a, 9))
        b = only(identical(:b, 1))
        @test sort([b, a]) == [a, b]
    end
    for (case, ub, lb) in [
        ("lb > ub", 2, 3),
        ("lb < 0", 2, -1),
        ("infinite ub", Inf, 0),
        ("fractional ub", 2.5, 0),
        ("fractional lb", 2, 0.5),
    ]
        @testset "[identical] rejects $case" begin
            @test_throws ArgumentError identical(:bins, ub; lb = lb)
        end
    end
end

function test_dml_identical_convexity_bounds()
    @testset "[identical] _convexity_bounds" begin
        @test Vertigo.DML._convexity_bounds(only(identical(:bins, 3; lb = 1))) == (1.0, 3.0)
        @test Vertigo.DML._convexity_bounds(2) == (1.0, 1.0)
        @test Vertigo.DML._convexity_bounds((:machine, "a")) == (1.0, 1.0)
    end
end

_convexity_set(decomp, ci) =
    with_typed_ci(c -> MOI.get(master_model(decomp), MOI.ConstraintSet(), c), ci)

function test_dml_identical_decomposition_bounds()
    big = only(identical(:big, 3; lb = 1))
    small = only(identical(:small, 2))
    K = [big, small]
    model = Model()
    @variable(model, x[k in K, i in 1:2], Bin)
    @constraint(model, cover[i in 1:2], sum(x[k, i] for k in K) >= 1)
    @objective(model, Min, sum(x))
    decomp, sp_map = @dantzig_wolfe model begin
        x[k, _] => subproblem(k)
        cover[_] => master()
    end

    for (block, lb, ub) in [(big, 1.0, 3.0), (small, 0.0, 2.0)]
        sp_id = sp_map[block]
        @testset "[identical] $block: bounds of the decomposition" begin
            @test convexity_bounds(decomp, sp_id) == (lb, ub)
        end
        @testset "[identical] $block: bounds of the master rows" begin
            @test _convexity_set(decomp, convexity_lb_ci(decomp, sp_id)) == MOI.GreaterThan(lb)
            @test _convexity_set(decomp, convexity_ub_ci(decomp, sp_id)) == MOI.LessThan(ub)
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
        @test MOI.get(sp_model(decomp, sp_map[only(K)]), MOI.ObjectiveFunctionType()) == saf
    end
end

function test_identical()
    test_dml_identical_index_set()
    test_dml_identical_convexity_bounds()
    test_dml_identical_decomposition_bounds()
    test_dml_single_variable_objective()
end
