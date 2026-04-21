# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

function test_build_cut_saf_empty_pool()
    @testset "[build_cut_saf] empty pool" begin
        ws, refs = build_insert_columns_context()
        model = master_model(ws.decomp)

        cut_coeffs = Dict{Tuple{Int,Int},Float64}((1, 1) => 3.0)
        cut = SeparatedCut(cut_coeffs, MOI.LessThan(1.0))

        saf = Vertigo.BranchCutPrice._build_cut_saf(
            ws.decomp, ws.pool, model, cut
        )

        @test isempty(saf.terms)
        @test saf.constant ≈ 0.0
    end
end

function test_build_cut_saf_single_column()
    @testset "[build_cut_saf] single column" begin
        ws, refs = build_insert_columns_context()
        model = master_model(ws.decomp)

        # Insert one column: z1=1.0
        col_var = MOI.add_variable(model)
        sol = Vertigo.Reformulation._SpSolution(
            PricingSubproblemId(1), 3.0, [(refs.z1, 1.0)]
        )
        record_column!(
            ws.pool, col_var, PricingSubproblemId(1), sol, 3.0
        )

        # Cut on original var (1,1) with coeff 3.0
        cut_coeffs = Dict{Tuple{Int,Int},Float64}((1, 1) => 3.0)
        cut = SeparatedCut(cut_coeffs, MOI.LessThan(1.0))

        saf = Vertigo.BranchCutPrice._build_cut_saf(
            ws.decomp, ws.pool, model, cut
        )

        @test length(saf.terms) == 1
        @test saf.terms[1].variable == col_var
        @test saf.terms[1].coefficient ≈ 3.0
    end
end

function test_build_cut_saf_two_columns()
    @testset "[build_cut_saf] two columns" begin
        ws, refs = build_insert_columns_context()
        model = master_model(ws.decomp)

        # col_a: z1=1.0 → maps to (1,1)
        col_a = MOI.add_variable(model)
        sol_a = Vertigo.Reformulation._SpSolution(
            PricingSubproblemId(1), 3.0, [(refs.z1, 1.0)]
        )
        record_column!(
            ws.pool, col_a, PricingSubproblemId(1), sol_a, 3.0
        )

        # col_b: z2=2.0 → maps to (1,2)
        col_b = MOI.add_variable(model)
        sol_b = Vertigo.Reformulation._SpSolution(
            PricingSubproblemId(1), 10.0, [(refs.z2, 2.0)]
        )
        record_column!(
            ws.pool, col_b, PricingSubproblemId(1), sol_b, 10.0
        )

        # Cut on (1,2) with coeff 5.0
        # col_a: z1→(1,1), no contribution
        # col_b: z2→(1,2), coeff = 5.0 * 2.0 = 10.0
        cut_coeffs = Dict{Tuple{Int,Int},Float64}((1, 2) => 5.0)
        cut = SeparatedCut(cut_coeffs, MOI.LessThan(1.0))

        saf = Vertigo.BranchCutPrice._build_cut_saf(
            ws.decomp, ws.pool, model, cut
        )

        @test length(saf.terms) == 1
        @test saf.terms[1].variable == col_b
        @test saf.terms[1].coefficient ≈ 10.0
    end
end

function test_build_cut_saf_mixed_sp_vars()
    @testset "[build_cut_saf] mixed SP variables" begin
        ws, refs = build_insert_columns_context()
        model = master_model(ws.decomp)

        # One column with z1=1.0, z2=2.0
        col_var = MOI.add_variable(model)
        sol = Vertigo.Reformulation._SpSolution(
            PricingSubproblemId(1), 13.0,
            [(refs.z1, 1.0), (refs.z2, 2.0)]
        )
        record_column!(
            ws.pool, col_var, PricingSubproblemId(1), sol, 13.0
        )

        # Cut coefficients: (1,1) → 3.0, (1,2) → 5.0
        # Expected: 3.0*1.0 + 5.0*2.0 = 13.0
        cut_coeffs = Dict{Tuple{Int,Int},Float64}(
            (1, 1) => 3.0, (1, 2) => 5.0
        )
        cut = SeparatedCut(cut_coeffs, MOI.LessThan(1.0))

        saf = Vertigo.BranchCutPrice._build_cut_saf(
            ws.decomp, ws.pool, model, cut
        )

        @test length(saf.terms) == 1
        @test saf.terms[1].variable == col_var
        @test saf.terms[1].coefficient ≈ 13.0
    end
end

function test_build_cut_saf_skips_invalid_column()
    @testset "[build_cut_saf] skips invalid column" begin
        ws, refs = build_insert_columns_context()
        model = master_model(ws.decomp)

        # Record a column with a variable NOT added to the model
        fake_var = MOI.VariableIndex(999)
        sol = Vertigo.Reformulation._SpSolution(
            PricingSubproblemId(1), 3.0, [(refs.z1, 1.0)]
        )
        record_column!(
            ws.pool, fake_var, PricingSubproblemId(1), sol, 3.0
        )

        cut_coeffs = Dict{Tuple{Int,Int},Float64}((1, 1) => 3.0)
        cut = SeparatedCut(cut_coeffs, MOI.LessThan(1.0))

        saf = Vertigo.BranchCutPrice._build_cut_saf(
            ws.decomp, ws.pool, model, cut
        )

        @test isempty(saf.terms)
    end
end

function test_build_cut_saf()
    test_build_cut_saf_empty_pool()
    test_build_cut_saf_single_column()
    test_build_cut_saf_two_columns()
    test_build_cut_saf_mixed_sp_vars()
    test_build_cut_saf_skips_invalid_column()
end
