# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ── Branch-and-price e2e tests on GAP class A ────────────────────────────

# (class, agents, jobs, optimal_value)
const GAP_A_BP_INSTANCES = [
    #('A', 5,  100, 1698),
    #('A', 5,  200, 3235),
    ('A', 10, 100, 1360),
    #('A', 10, 200, 2623),
    ('A', 20, 100, 1158),
    #('A', 20, 200, 2339)
]

function test_bp_gap_a_instances()
    for (class, agents, jobs, optimal) in GAP_A_BP_INSTANCES
        label = "$(uppercase(class)) $(agents)×$(jobs)"
        @testset "[bp][$(label)] optimal = $(optimal)" begin
            filepath = get_gap_instance_path(class, agents, jobs)
            inst = parse_gap_file(filepath)
            decomp = build_gap_decomp(inst)
            dot_path = "$class-$agents-$jobs.dot"
            bcp_ws = BranchCutPriceWorkspace(
                decomp,
                BranchCutPriceConfig(
                    cg_config=ColGenConfig(smoothing_alpha=0.5),
                    node_limit=5_000,
                    dot_file=dot_path,
                ),
            )
            output = run_branch_and_price(bcp_ws)

            @test output.status in (:optimal, :node_limit)

            # Dual bound must be valid (≤ optimal for min)
            @test output.best_dual_bound <= optimal + 1e-4

            if !isnothing(output.incumbent)
                # Incumbent is a valid upper bound
                @test output.incumbent.obj_value >= optimal - 1e-4

                # Dual bound ≤ incumbent (valid relaxation)
                @test output.best_dual_bound <= output.incumbent.obj_value + 1e-4
            end

            if output.status == :optimal
                @test !isnothing(output.incumbent)
                @test abs(output.incumbent.obj_value - optimal) <= 1e-4
            end

            # Dot file structural validity
            @test isfile(dot_path)
            dot_content = read(dot_path, String)
            @test startswith(dot_content, "digraph BCP_Tree {")
            @test endswith(strip(dot_content), "}")
            @test occursin(r"n\d+ \[label=", dot_content)
        end
    end
end

# Column generation can meet an integer master solution before it converges. The node must
# then be branched, not closed: here the root LP is 63 while the first integer solution
# found by column generation costs 65 (issue 76).
function test_bp_ip_incumbent_before_convergence()
    @testset "[bp] an integer solution found during column generation does not close the node" begin
        inst = random_gap_instance(4, 12; seed=1)
        optimal = 63.0   # flat MIP optimum
        bcp_ws = BranchCutPriceWorkspace(
            build_gap_decomp(inst), BranchCutPriceConfig(node_limit=2_000)
        )
        output = run_branch_and_price(bcp_ws)
        @test output.status == :optimal
        @test output.root_lp_value ≈ optimal atol = 1e-4
        @test output.incumbent.obj_value ≈ optimal atol = 1e-4
        @test output.best_dual_bound <= optimal + 1e-4
    end
end

# A branching constraint is re-created, when its node is entered, from the columns that
# existed when it was created. A column generated since, in a node where the constraint is
# not active, must get its coefficient, or the pricing finds the column again and it is
# refused as a duplicate (issue 88).
function test_bp_branching_constraint_coefficients()
    @testset "[bp] a column generated elsewhere gets the coefficient of a re-created branching constraint" begin
        space = BranchCutPriceWorkspace(
            build_gap_decomp(random_gap_instance(2, 5; seed=1)), BranchCutPriceConfig()
        )
        ws = space.ws
        decomp = ws.decomp
        root = Vertigo.TreeSearch.new_root(space)

        # Branch on an original variable.
        sp_id = first(Vertigo.Decomposition.subproblem_ids(decomp))
        sp_var = first(Vertigo.Decomposition.subproblem_variables(decomp, sp_id))
        orig_var = Vertigo.Decomposition.mapped_original_var(decomp, sp_id, sp_var)
        (down, up), cut_info = Vertigo.BranchCutPrice.create_branching_children(
            space.id_counter, root, orig_var, 0.5, ws, root.dual_bound, space.cut_tracker
        )
        for (cut_id, ov) in cut_info
            space.branching_cut_info[cut_id] = ov
        end

        # A column that uses the variable, generated while the constraint is not active.
        sol = Vertigo.Decomposition._SpSolution(sp_id, -1.0, [(sp_var, 1.0)])
        col_var = Vertigo.ColGen._insert_column!(
            ws,
            Vertigo.ColGen.Phase2(),
            Vertigo.ColGen.PricingPrimalSolution(sp_id, sol, true),
        )
        @test !isnothing(col_var)

        # Entering the up node re-creates the constraint without the column.
        Vertigo.TreeSearch.transition!(space, root, up)
        Vertigo.BranchCutPrice._rebuild_branching_constraints!(space)

        model = Vertigo.Branching.bp_master_model(ws)
        (bc,) = Vertigo.Branching.bp_branching_constraints(ws)
        f = Vertigo.with_typed_ci(bc.constraint_index) do ci
            return MOI.get(model, MOI.ConstraintFunction(), ci)
        end
        in_master = Dict(t.variable => t.coefficient for t in f.terms)
        expected = Vertigo.Decomposition.compute_branching_column_coefficient(
            decomp, orig_var, sp_id, sol
        )
        @test expected != 0.0
        @test get(in_master, col_var, 0.0) == expected
    end
end
