# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.Branching: find_fractional_variables,
    BranchingCandidate, MostFractionalRule, LeastFractionalRule,
    select_candidates, MostFractionalBranching,
    select_branching_variable, bp_master_model,
    most_fractional_original_variable,
    BranchingResult, branching_ok, all_integral
using Vertigo.BranchCutPrice: BPSpace
using Vertigo.Reformulation: columns, column_nonzero_entries,
    column_sp_id, mapped_original_var

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

"""
Build a primal dict where all λ = 0 (no column active).
Projects to x = 0 for every original variable → all integral.
"""
function _integral_primal(ctx)
    return Dict{MOI.VariableIndex,Float64}()
end

"""
Build a primal dict with hand-chosen λ values that produce
known fractional original variables.

Strategy: pick two columns from different subproblems, give them
λ values such that the projected x has controlled fractionalities.
Returns `(primal, expected)` where `expected` is a vector of
`(orig_var, x_val, fractionality)` sorted by fractionality desc.
"""
function _fractional_primal(ctx)
    decomp = ctx.decomp
    pool = ctx.pool

    # Collect up to 2 columns from different subproblems
    col_entries = collect(columns(pool))
    length(col_entries) < 2 && error(
        "pool must have >= 2 columns for this test"
    )

    col_var_a, rec_a = col_entries[1]
    col_var_b, rec_b = col_entries[2]

    # λ_a = 0.3, λ_b = 0.7 — both fractional
    λ_a = 0.3
    λ_b = 0.7

    primal = Dict{MOI.VariableIndex,Float64}(
        col_var_a => λ_a,
        col_var_b => λ_b,
    )

    # Compute expected projection: x[orig] = Σ z_val * λ_val
    x_values = Dict{Any,Float64}()
    for (col_var, rec, λ) in [
        (col_var_a, rec_a, λ_a),
        (col_var_b, rec_b, λ_b),
    ]
        sp_id = column_sp_id(rec)
        for (sp_var, z_val) in column_nonzero_entries(rec)
            orig = mapped_original_var(decomp, sp_id, sp_var)
            orig === nothing && continue
            x_values[orig] = get(x_values, orig, 0.0) + z_val * λ
        end
    end

    # Keep only fractional entries
    tol = 1e-6
    expected = Tuple{Any,Float64,Float64}[]
    for (orig, x_val) in x_values
        frac_part = x_val - floor(x_val)
        (frac_part < tol || frac_part > 1.0 - tol) && continue
        fractionality = min(frac_part, 1.0 - frac_part)
        push!(expected, (orig, x_val, fractionality))
    end
    sort!(expected; by=e -> e[3], rev=true)

    return primal, expected
end

# ────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────

function test_find_fractional_all_integral()
    @testset "[find_fractional_variables] all integral" begin
        inst = random_gap_instance(2, 5; seed=42)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)

        primal = _integral_primal(ctx)
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        @test isempty(candidates)
    end
end

function test_find_fractional_with_known_solution()
    @testset "[find_fractional_variables] detects fractional with forged primal" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)

        primal, expected = _fractional_primal(ctx)
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )

        @test length(candidates) == length(expected)

        for (i, (orig, x_val, frac)) in enumerate(expected)
            @test candidates[i].orig_var == orig
            @test candidates[i].value ≈ x_val atol=1e-10
            @test candidates[i].fractionality ≈ frac atol=1e-10
            @test candidates[i].floor_val ≈ floor(x_val) atol=1e-10
            @test candidates[i].ceil_val ≈ ceil(x_val) atol=1e-10
        end
    end
end

function test_select_candidates_most_fractional()
    @testset "[branching_rules] MostFractionalRule ordering" begin
        c1 = BranchingCandidate(1, 1.3, 1.0, 2.0, 0.3)
        c2 = BranchingCandidate(2, 2.5, 2.0, 3.0, 0.5)
        c3 = BranchingCandidate(3, 3.1, 3.0, 4.0, 0.1)
        candidates = [c2, c1, c3]
        result = select_candidates(
            MostFractionalRule(), candidates, 2
        )
        @test length(result) == 2
        @test result[1].fractionality ≈ 0.5
        @test result[2].fractionality ≈ 0.3
    end
end

function test_select_candidates_least_fractional()
    @testset "[branching_rules] LeastFractionalRule ordering" begin
        c1 = BranchingCandidate(1, 1.3, 1.0, 2.0, 0.3)
        c2 = BranchingCandidate(2, 2.5, 2.0, 3.0, 0.5)
        c3 = BranchingCandidate(3, 3.1, 3.0, 4.0, 0.1)
        candidates = [c2, c1, c3]
        result = select_candidates(
            LeastFractionalRule(), candidates, 2
        )
        @test length(result) == 2
        @test result[1].fractionality ≈ 0.1
        @test result[2].fractionality ≈ 0.3
    end
end

function test_select_candidates_truncation()
    @testset "[branching_rules] max_candidates truncation" begin
        cs = [
            BranchingCandidate(
                i, Float64(i) + 0.5, Float64(i),
                Float64(i) + 1.0, 0.5
            ) for i in 1:10
        ]
        result = select_candidates(MostFractionalRule(), cs, 3)
        @test length(result) == 3
    end
end

function test_most_fractional_branching_delegates()
    @testset "[branching_strategy] MostFractionalBranching delegates" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)

        primal, expected = _fractional_primal(ctx)
        # The most fractional variable is the first in expected
        most_frac_orig = expected[1][1]
        most_frac_val = expected[1][2]

        space = BPSpace(ctx; node_limit=1)
        result = select_branching_variable(
            MostFractionalBranching(), space, nothing, primal
        )
        @test result.status == branching_ok
        @test result.orig_var == most_frac_orig
        @test result.value ≈ most_frac_val atol=1e-10
    end
end

# ────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────

function test_branching_strategy()
    test_find_fractional_all_integral()
    test_find_fractional_with_known_solution()
    test_select_candidates_most_fractional()
    test_select_candidates_least_fractional()
    test_select_candidates_truncation()
    test_most_fractional_branching_delegates()
end
