# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ── Global dual bound e2e tests on GAP class A ───────────────────────────

# Branches on the original variable of the first subproblem variable,
# whatever the LP solution: whether the root LP solution is fractional
# depends on which optimal solution the solver returns.
struct BranchOnFirstVariable <: Vertigo.Branching.AbstractBranchingStrategy end

function Vertigo.Branching.select_branching_variable(
    ::BranchOnFirstVariable, space, node, primal_values
)
    decomp = space.ws.decomp
    sp_id = first(Vertigo.Reformulation.subproblem_ids(decomp))
    sp_var = first(Vertigo.Reformulation.subproblem_variables(decomp, sp_id))
    orig_var = Vertigo.Reformulation.mapped_original_var(decomp, sp_id, sp_var)
    return Vertigo.Branching.BranchingResult(orig_var, 0.5)
end

function test_dual_bound_e2e()
    @testset "[dual bound] the root bound is known as soon as the root is evaluated" begin
        inst = parse_gap_file(get_gap_instance_path('A', 10, 100))
        space = BranchCutPriceWorkspace(
            build_gap_decomp(inst),
            BranchCutPriceConfig(colgen=ColGenConfig(smoothing_alpha=0.5)),
        )
        root = Vertigo.TreeSearch.new_root(space)
        Vertigo.TreeSearch.evaluate!(BPEvaluator(), space, root)
        @test isfinite(Vertigo.TreeSearch.ts_best_dual_bound(space))
        @test Vertigo.TreeSearch.ts_best_dual_bound(space) <= 1360 + 1e-4
    end

    @testset "[dual bound] a node stopped by the time limit keeps its parent's bound" begin
        inst = parse_gap_file(get_gap_instance_path('A', 10, 100))
        space = BranchCutPriceWorkspace(
            build_gap_decomp(inst),
            BranchCutPriceConfig(colgen=ColGenConfig(smoothing_alpha=0.0)),
        )
        root = Vertigo.TreeSearch.new_root(space)
        Vertigo.TreeSearch.evaluate!(BPEvaluator(), space, root)
        root_bound = Vertigo.TreeSearch.ts_best_dual_bound(space)
        children = Vertigo.TreeSearch.branch!(space, root)
        child = first(children)
        Vertigo.TreeSearch.transition!(space, root, child)
        Vertigo.ColGen.set_deadline!(space.ws, time() - 1.0)
        status = Vertigo.TreeSearch.evaluate!(BPEvaluator(), space, child)
        @test status == Vertigo.TreeSearch.CUTOFF
        @test child.dual_bound >= root_bound - 1e-6
        @test Vertigo.TreeSearch.ts_best_dual_bound(space) >= root_bound - 1e-6
    end

    @testset "[dual bound] children inherit the node bound, not the column generation one" begin
        inst = parse_gap_file(get_gap_instance_path('A', 10, 100))
        space = BranchCutPriceWorkspace(
            build_gap_decomp(inst),
            BranchCutPriceConfig(branching_strategy=BranchOnFirstVariable()),
        )
        root = Vertigo.TreeSearch.new_root(space)
        Vertigo.TreeSearch.evaluate!(BPEvaluator(), space, root)
        # Stands for a bound inherited from an ancestor, tighter than the
        # one this column generation found (issue #86).
        root.dual_bound += 0.5
        children = Vertigo.TreeSearch.branch!(space, root)
        @test length(children) == 2
        @test children[1].dual_bound ≈ root.dual_bound atol = 1e-10
        @test children[2].dual_bound ≈ root.dual_bound atol = 1e-10
    end

    @testset "[dual bound] a closed tree has the incumbent as dual bound" begin
        inst = parse_gap_file(get_gap_instance_path('A', 20, 100))
        output = run_branch_and_price(
            BranchCutPriceWorkspace(build_gap_decomp(inst), BranchCutPriceConfig())
        )
        @test output.status == :optimal
        @test output.best_dual_bound ≈ output.incumbent.obj_value
    end

    @testset "[dual bound] an infeasible problem has +Inf as dual bound" begin
        output = run_branch_and_price(
            BranchCutPriceWorkspace(
                build_gap_decomp(gap_infeasible_master()), BranchCutPriceConfig()
            )
        )
        @test output.status == :infeasible
        @test output.best_dual_bound == Inf
    end
end
