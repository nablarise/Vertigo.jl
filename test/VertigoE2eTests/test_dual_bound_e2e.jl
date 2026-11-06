# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ── Global dual bound e2e tests on GAP class A ───────────────────────────

# Branches on the original variable of the first subproblem variable, whatever the solution.
struct BranchOnFirstVariable <: Vertigo.Branching.AbstractBranchingStrategy end

function Vertigo.Branching.select_branching_variable(
    ::BranchOnFirstVariable, space, node, primal_values
)
    decomp = space.ws.decomp
    sp_id = first(Vertigo.Decomposition.subproblem_ids(decomp))
    sp_var = first(Vertigo.Decomposition.subproblem_variables(decomp, sp_id))
    orig_var = Vertigo.Decomposition.mapped_original_var(decomp, sp_id, sp_var)
    return Vertigo.Branching.BranchingResult(orig_var, 0.5)
end

function test_dual_bound_e2e()
    @testset "[dual bound] the root bound is known as soon as the root is evaluated" begin
        inst = parse_gap_file(get_gap_instance_path('A', 10, 100))
        space = BranchCutPriceWorkspace(
            build_gap_decomp(inst),
            BranchCutPriceConfig(cg_config=ColGenConfig(smoothing_alpha=0.5)),
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
            BranchCutPriceConfig(cg_config=ColGenConfig(smoothing_alpha=0.0)),
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

    @testset "[dual bound] the children inherit the dual bound of the node" begin
        # A random GAP root is rarely fractional, so the branching is forced on a variable.
        space = BranchCutPriceWorkspace(
            build_gap_decomp(random_gap_instance(2, 5; seed=1)),
            BranchCutPriceConfig(branching_strategy=BranchOnFirstVariable()),
        )
        root = Vertigo.TreeSearch.new_root(space)
        Vertigo.TreeSearch.evaluate!(BPEvaluator(), space, root)
        MOI.optimize!(space.backend)   # the LP solution `branch!` reads is up to date
        # A bound tighter than the one found by column generation, as the bound a node
        # inherits can be: the children must start from it, not from the weaker one.
        root.dual_bound += 1.0
        children = Vertigo.TreeSearch.branch!(space, root)
        @test length(children) == 2
        @test all(child -> child.dual_bound == root.dual_bound, children)
    end

    @testset "[dual bound] a closed tree has the incumbent as dual bound" begin
        inst = parse_gap_file(get_gap_instance_path('A', 20, 100))
        output = run_branch_and_price(
            BranchCutPriceWorkspace(build_gap_decomp(inst), BranchCutPriceConfig())
        )
        @test output.status == :optimal
        @test output.best_dual_bound ≈ output.incumbent.obj_value
    end
end
