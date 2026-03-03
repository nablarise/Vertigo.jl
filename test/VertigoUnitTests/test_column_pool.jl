# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ──────────────────────────────────────────────────────────────────
# SpSolution fingerprint tests
# ──────────────────────────────────────────────────────────────────

function test_fingerprint_same_entries()
    @testset "[fingerprint] same entries produce same fingerprint" begin
        z1 = MOI.VariableIndex(1)
        z2 = MOI.VariableIndex(2)

        sol_a = SpSolution(1, 3.0, [(z1, 1.0), (z2, 2.0)])
        sol_b = SpSolution(1, 99.0, [(z1, 1.0), (z2, 2.0)])

        @test sol_a.fingerprint == sol_b.fingerprint
    end
end

function test_fingerprint_different_values()
    @testset "[fingerprint] different values produce different fingerprints" begin
        z1 = MOI.VariableIndex(1)

        sol_a = SpSolution(1, 3.0, [(z1, 1.0)])
        sol_b = SpSolution(1, 3.0, [(z1, 2.0)])

        @test sol_a.fingerprint != sol_b.fingerprint
    end
end

function test_fingerprint_zero_entries_filtered()
    @testset "[fingerprint] zero entries are filtered out" begin
        z1 = MOI.VariableIndex(1)
        z2 = MOI.VariableIndex(2)

        sol_a = SpSolution(1, 3.0, [(z1, 1.0)])
        sol_b = SpSolution(1, 3.0, [(z1, 1.0), (z2, 0.0)])

        @test sol_a.fingerprint == sol_b.fingerprint
        @test length(sol_a.entries) == 1
        @test length(sol_b.entries) == 1
    end
end

function test_fingerprint_order_independent()
    @testset "[fingerprint] entry order does not matter" begin
        z1 = MOI.VariableIndex(1)
        z2 = MOI.VariableIndex(2)

        sol_a = SpSolution(1, 3.0, [(z1, 1.0), (z2, 2.0)])
        sol_b = SpSolution(1, 3.0, [(z2, 2.0), (z1, 1.0)])

        @test sol_a.fingerprint == sol_b.fingerprint
    end
end

# ──────────────────────────────────────────────────────────────────
# ColumnPool tests
# ──────────────────────────────────────────────────────────────────

function test_column_pool_record_and_retrieve()
    @testset "[column_pool] record and retrieve" begin
        pool = ColumnPool{MOI.VariableIndex,Int,MOI.VariableIndex}()

        z1 = MOI.VariableIndex(1)
        sol = SpSolution(1, 3.0, [(z1, 1.0)])
        col_var = MOI.VariableIndex(100)

        record_column!(pool, col_var, 1, sol, 5.0)

        @test length(collect(columns(pool))) == 1
        @test get_column_solution(pool, col_var) === sol
        @test get_column_sp_id(pool, col_var) == 1
        @test get_column_cost(pool, col_var) ≈ 5.0
    end
end

function test_column_pool_get_column_solution_missing()
    @testset "[column_pool] get_column_solution returns nothing" begin
        pool = ColumnPool{MOI.VariableIndex,Int,MOI.VariableIndex}()

        @test get_column_solution(pool, MOI.VariableIndex(999)) === nothing
    end
end

function test_column_pool_columns_for_subproblem()
    @testset "[column_pool] columns_for_subproblem" begin
        pool = ColumnPool{MOI.VariableIndex,Int,MOI.VariableIndex}()

        z1 = MOI.VariableIndex(1)
        z2 = MOI.VariableIndex(2)
        sol_a = SpSolution(1, 3.0, [(z1, 1.0)])
        sol_b = SpSolution(1, 7.0, [(z2, 2.0)])
        sol_c = SpSolution(2, 4.0, [(z1, 1.0)])

        col_a = MOI.VariableIndex(100)
        col_b = MOI.VariableIndex(101)
        col_c = MOI.VariableIndex(102)

        record_column!(pool, col_a, 1, sol_a, 3.0)
        record_column!(pool, col_b, 1, sol_b, 7.0)
        record_column!(pool, col_c, 2, sol_c, 4.0)

        sp1_cols = collect(columns_for_subproblem(pool, 1))
        @test length(sp1_cols) == 2
        sp1_costs = sort([c[3] for c in sp1_cols])
        @test sp1_costs[1] ≈ 3.0
        @test sp1_costs[2] ≈ 7.0

        sp2_cols = collect(columns_for_subproblem(pool, 2))
        @test length(sp2_cols) == 1
        @test sp2_cols[1][3] ≈ 4.0

        # Non-existent subproblem returns empty.
        @test isempty(collect(columns_for_subproblem(pool, 99)))
    end
end

function test_column_pool_columns_iterator()
    @testset "[column_pool] columns iterator" begin
        pool = ColumnPool{MOI.VariableIndex,Int,MOI.VariableIndex}()

        z1 = MOI.VariableIndex(1)
        z2 = MOI.VariableIndex(2)
        sol_a = SpSolution(1, 3.0, [(z1, 1.0)])
        sol_b = SpSolution(2, 7.0, [(z2, 2.0)])

        col_a = MOI.VariableIndex(100)
        col_b = MOI.VariableIndex(101)

        record_column!(pool, col_a, 1, sol_a, 3.0)
        record_column!(pool, col_b, 2, sol_b, 7.0)

        all_cols = collect(columns(pool))
        @test length(all_cols) == 2

        costs = sort([c[4] for c in all_cols])
        @test costs[1] ≈ 3.0
        @test costs[2] ≈ 7.0
    end
end

function test_column_pool_has_column_duplicate()
    @testset "[column_pool] has_column detects duplicate" begin
        pool = ColumnPool{MOI.VariableIndex,Int,MOI.VariableIndex}()

        z1 = MOI.VariableIndex(1)
        sol = SpSolution(1, 3.0, [(z1, 1.0)])
        col_var = MOI.VariableIndex(100)

        @test !has_column(pool, 1, sol)

        record_column!(pool, col_var, 1, sol, 5.0)

        @test has_column(pool, 1, sol)

        # Same entries, rebuilt — same fingerprint.
        sol_dup = SpSolution(1, 99.0, [(z1, 1.0)])
        @test has_column(pool, 1, sol_dup)
    end
end

function test_column_pool_has_column_different_values()
    @testset "[column_pool] has_column distinguishes different values" begin
        pool = ColumnPool{MOI.VariableIndex,Int,MOI.VariableIndex}()

        z1 = MOI.VariableIndex(1)
        sol_a = SpSolution(1, 3.0, [(z1, 1.0)])
        col_var = MOI.VariableIndex(100)

        record_column!(pool, col_var, 1, sol_a, 5.0)

        # Same variable, different value.
        sol_b = SpSolution(1, 3.0, [(z1, 2.0)])
        @test !has_column(pool, 1, sol_b)
    end
end

function test_column_pool_has_column_different_subproblem()
    @testset "[column_pool] has_column is subproblem-scoped" begin
        pool = ColumnPool{MOI.VariableIndex,Int,MOI.VariableIndex}()

        z1 = MOI.VariableIndex(1)
        sol = SpSolution(1, 3.0, [(z1, 1.0)])
        col_var = MOI.VariableIndex(100)

        record_column!(pool, col_var, 1, sol, 5.0)

        # Same fingerprint but different subproblem.
        sol2 = SpSolution(2, 3.0, [(z1, 1.0)])
        @test !has_column(pool, 2, sol2)
    end
end

function test_column_pool_empty()
    @testset "[column_pool] empty pool" begin
        pool = ColumnPool{MOI.VariableIndex,Int,MOI.VariableIndex}()

        @test isempty(collect(columns(pool)))

        sol = SpSolution(1, 0.0, [(MOI.VariableIndex(1), 1.0)])
        @test !has_column(pool, 1, sol)
    end
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

function test_column_pool()
    test_fingerprint_same_entries()
    test_fingerprint_different_values()
    test_fingerprint_zero_entries_filtered()
    test_fingerprint_order_independent()
    test_column_pool_record_and_retrieve()
    test_column_pool_get_column_solution_missing()
    test_column_pool_columns_for_subproblem()
    test_column_pool_columns_iterator()
    test_column_pool_has_column_duplicate()
    test_column_pool_has_column_different_values()
    test_column_pool_has_column_different_subproblem()
    test_column_pool_empty()
end
