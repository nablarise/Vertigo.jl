# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

function test_branch_and_price()
    @testset "[branch_and_price] branch after pricing adds a column on the last iteration (#73)" begin
        inst = random_gap_instance(2, 5; seed=10)
        # The root LP is fractional. Stop CG immediately after pricing so
        # the master contains columns absent from the last LP solution.
        config = BranchCutPriceConfig(
            colgen = ColGenConfig(max_cg_iterations = 1),
            rmp_heuristic = false,
            node_limit = 1
        )
        space = BranchCutPriceWorkspace(build_gap_decomp(inst), config)
        output = run_branch_and_price(space)
        @test output.status == :node_limit
        @test length(space.open_node_bounds) == 2
        @test isfinite(output.best_dual_bound)
        @test output.best_dual_bound <= output.root_lp_value + 1e-6
    end

    @testset "[branch_and_price] small GAP finds integer solution" begin
        inst = random_gap_instance(2, 4; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp, BranchCutPriceConfig(node_limit = 100)
        )
        output = run_branch_and_price(bcp_ws)
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
        @test output.nodes_explored >= 1
        @test !isnothing(output.root_lp_value)
        @test output.root_lp_value <=
              output.incumbent.obj_value + 1e-6
    end

    @testset "[branch_and_price] respects node limit" begin
        inst = random_gap_instance(3, 8; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp, BranchCutPriceConfig(node_limit = 5)
        )
        output = run_branch_and_price(bcp_ws)
        @test output.nodes_explored <= 5
    end

    @testset "[branch_and_price] fully explored tree reports :optimal" begin
        # Regression for #63: when every node is processed, the dual
        # bound must not be reset to -Inf and the status must be
        # :optimal rather than :node_limit.
        #
        # Uses the toy GAP instance from `examples/gap/instances/toy.json`
        # (3 machines, 6 tasks; optimum = 160 by enumerating the 3^6
        # assignments: task 1 on machine 1, tasks 2-3 on machine 2,
        # tasks 4-6 on machine 3). The restricted-master
        # IP heuristic is disabled so the incumbent comes from the LP/CG
        # path and the empty-`open_node_bounds` branch is exercised.
        cost = Float64[
            10 12 13 14 15 17;
            21 22 23 24 25 28;
            31 32 33 34 35 36
        ]
        weight = Float64[
            6 6 6 6 6 6;
            4 4 4 4 4 4;
            2 2 2 2 2 2
        ]
        capacity = Float64[10, 10, 10]
        inst = GAPInstance(3, 6, cost, weight, capacity)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp,
            BranchCutPriceConfig(node_limit = 1000, rmp_heuristic = false)
        )
        output = run_branch_and_price(bcp_ws)
        @test output.status == :optimal
        @test output.nodes_explored < 1000
        @test !isnothing(output.incumbent)
        @test output.incumbent.obj_value ≈ 160.0 atol = 1e-6
        @test output.best_dual_bound ≈ output.incumbent.obj_value atol = 1e-6
    end

    @testset "[branch_and_price] dual bound is valid" begin
        inst = random_gap_instance(2, 5; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp, BranchCutPriceConfig(node_limit = 50)
        )
        output = run_branch_and_price(bcp_ws)
        if !isnothing(output.incumbent)
            @test output.best_dual_bound <=
                  output.incumbent.obj_value + 1e-6
        end
    end

    @testset "[branch_and_price] logger runs without error" begin
        inst = random_gap_instance(2, 4; seed=42)
        decomp = build_gap_decomp(inst)
        bcp_ws = BranchCutPriceWorkspace(
            decomp,
            BranchCutPriceConfig(node_limit = 100, log_level = 1)
        )
        output = redirect_stdout(devnull) do
            run_branch_and_price(bcp_ws)
        end
        @test output.status in (:optimal, :node_limit)
        @test output.nodes_explored >= 1
    end

    @testset "[branch_and_price] logger result matches non-logger" begin
        inst = random_gap_instance(2, 4; seed=42)
        decomp1 = build_gap_decomp(inst)
        out1 = run_branch_and_price(
            BranchCutPriceWorkspace(
                decomp1, BranchCutPriceConfig(node_limit = 50)
            )
        )

        decomp2 = build_gap_decomp(inst)
        out2 = redirect_stdout(devnull) do
            run_branch_and_price(BranchCutPriceWorkspace(
                decomp2,
                BranchCutPriceConfig(node_limit = 50, log_level = 1)
            ))
        end

        @test out1.status == out2.status
        @test out1.nodes_explored == out2.nodes_explored
    end

    # Issue #76: column generation can meet an integer master solution
    # before it converges. No solver runs here: the node bound and the
    # integer solution are set by hand on a 2 machines x 2 tasks GAP.
    function root_with_bound(bound)
        inst = GAPInstance(
            2, 2, [1.0 2.0; 3.0 4.0], [1.0 1.0; 1.0 1.0], [2.0, 2.0]
        )
        space = BranchCutPriceWorkspace(
            build_gap_decomp(inst), BranchCutPriceConfig()
        )
        root = Vertigo.TreeSearch.new_root(space)
        root.dual_bound = bound
        space.open_node_bounds[root.id] = bound
        return space, root
    end
    ip_sol(value) = Vertigo.ColGen.MasterIpPrimalSol(
        value, Tuple{MOI.VariableIndex,Int}[]
    )

    @testset "[branch_and_price] an integer solution above the node bound keeps the node open" begin
        space, root = root_with_bound(155.0)
        closed = Vertigo.BranchCutPrice._close_on_ip_incumbent!(
            space, root, ip_sol(160.0)
        )
        @test !closed
        @test space.incumbent.obj_value ≈ 160.0 atol = 1e-10
        @test haskey(space.open_node_bounds, root.id)
    end

    @testset "[branch_and_price] an integer solution at the node bound closes the node" begin
        space, root = root_with_bound(160.0)
        closed = Vertigo.BranchCutPrice._close_on_ip_incumbent!(
            space, root, ip_sol(160.0)
        )
        @test closed
        @test space.incumbent.obj_value ≈ 160.0 atol = 1e-10
        @test !haskey(space.open_node_bounds, root.id)
        @test space.best_dual_bound ≈ 160.0 atol = 1e-10
    end

    # Issue #128: a node whose projection is integral while its columns are
    # not is unbranchable. No solver runs: on the 2 x 2 GAP of
    # `root_with_bound`, machine 1 gets the columns {task 1}, {tasks 1, 2}
    # and {task 2}, and the master LP values are set by hand. With the
    # three at 0.5, x[1,1] = x[1,2] = 0.5 + 0.5 = 1 while no column is
    # integral.
    function unbranchable_root(bound)
        space, root = root_with_bound(bound)
        sp_id = PricingSubproblemId(1)
        z1, z2 = subproblem_variables(space.ws.decomp, sp_id)
        task_1, both, task_2 = MOI.VariableIndex.((901, 902, 903))
        for (col_var, entries) in (
            task_1 => [(z1, 1.0)],
            both => [(z1, 1.0), (z2, 1.0)],
            task_2 => [(z2, 1.0)]
        )
            sol = Vertigo.Reformulation._SpSolution(sp_id, 0.0, entries)
            record_column!(space.ws.pool, col_var, sp_id, sol, 0.0)
        end
        return space, root, (task_1, both, task_2)
    end
    function set_master_lp_values!(root, values)
        root.user_data.cg_output = Vertigo.ColGen.ColGenOutput(
            Vertigo.ColGen.optimal, root.dual_bound, root.dual_bound, nothing,
            MasterPrimalSolution(PrimalMoiSolution(root.dual_bound, values))
        )
    end

    @testset "[branch_and_price] fractional columns with an integral projection leave the node unbranchable" begin
        space, root, (task_1, both, task_2) = unbranchable_root(150.0)
        set_master_lp_values!(root, Dict(task_1 => 0.5, both => 0.5, task_2 => 0.5))
        children = TreeSearch.branch!(space, root)
        output = TreeSearch.output(space)
        @test output.status == :branching_incomplete
        @test isempty(children)
        @test !haskey(space.open_node_bounds, root.id)
        @test TreeSearch.ts_open_node_count(space) == 0
        @test space.unbranchable_node_bounds[root.id] ≈ 150.0 atol = 1e-10
        @test output.best_dual_bound ≈ 150.0 atol = 1e-10
    end

    @testset "[branch_and_price] integral columns close the node" begin
        space, root, (task_1, both, task_2) = unbranchable_root(150.0)
        set_master_lp_values!(root, Dict(task_1 => 1.0, both => 0.0, task_2 => 1.0))
        children = TreeSearch.branch!(space, root)
        @test isempty(children)
        @test !haskey(space.open_node_bounds, root.id)
        @test isempty(space.unbranchable_node_bounds)
    end

    @testset "[branch_and_price] a fractional integer pure master variable leaves the node unbranchable" begin
        space, root, (task_1, both, task_2) = unbranchable_root(150.0)
        y = _add_test_pure_master_var!(space.ws, 904, 1.0, true)
        set_master_lp_values!(
            root, Dict(task_1 => 1.0, both => 0.0, task_2 => 1.0, y => 0.5)
        )
        children = TreeSearch.branch!(space, root)
        @test isempty(children)
        @test haskey(space.unbranchable_node_bounds, root.id)
    end

    @testset "[branch_and_price] an incumbent at the bound of an unbranchable node prunes it" begin
        space, root, (task_1, both, task_2) = unbranchable_root(150.0)
        set_master_lp_values!(root, Dict(task_1 => 0.5, both => 0.5, task_2 => 0.5))
        TreeSearch.branch!(space, root)
        space.incumbent = ip_sol(150.0)
        Vertigo.BranchCutPrice._recompute_global_dual_bound!(space)
        output = TreeSearch.output(space)
        @test output.status == :optimal
        @test isempty(space.unbranchable_node_bounds)
        @test output.best_dual_bound ≈ 150.0 atol = 1e-10
    end

    @testset "[branch_and_price] an incumbent above the bound of an unbranchable node does not prune it" begin
        space, root, (task_1, both, task_2) = unbranchable_root(150.0)
        set_master_lp_values!(root, Dict(task_1 => 0.5, both => 0.5, task_2 => 0.5))
        TreeSearch.branch!(space, root)
        space.incumbent = ip_sol(160.0)
        Vertigo.BranchCutPrice._recompute_global_dual_bound!(space)
        output = TreeSearch.output(space)
        @test output.status == :branching_incomplete
        @test haskey(space.unbranchable_node_bounds, root.id)
        @test output.best_dual_bound ≈ 150.0 atol = 1e-10
    end

    @testset "[branch_and_price] the global dual bound covers open and unbranchable nodes" begin
        space, root = root_with_bound(155.0)
        space.unbranchable_node_bounds[root.id + 1] = 150.0
        Vertigo.BranchCutPrice._recompute_global_dual_bound!(space)
        # min(155 open, 150 unbranchable)
        @test space.best_dual_bound ≈ 150.0 atol = 1e-10
    end

    @testset "[branch_and_price] the dot node label shows the node bound" begin
        io = IOBuffer()
        Vertigo.BranchCutPrice._write_dot_node(
            io, (id = 3, dual_bound = 410.5),
            Vertigo.BranchCutPrice.DotBranched, nothing
        )
        @test occursin("DB = 410.50", String(take!(io)))
    end

    # Issue #88: a branching constraint is re-created, when its node is
    # entered, from the columns that existed when it was created.
    @testset "[branch_and_price] a column generated elsewhere gets the coefficient of a re-created branching constraint" begin
        inst = GAPInstance(
            2, 2, [1.0 2.0; 3.0 4.0], [1.0 1.0; 1.0 1.0], [2.0, 2.0]
        )
        space = BranchCutPriceWorkspace(
            build_gap_decomp(inst), BranchCutPriceConfig()
        )
        ws = space.ws
        decomp = ws.decomp
        root = Vertigo.TreeSearch.new_root(space)

        # Branch while the pool is empty: the constraints hold no column.
        sp_id = PricingSubproblemId(1)
        sp_var = first(Vertigo.Reformulation.subproblem_variables(decomp, sp_id))
        orig_var = Vertigo.Reformulation.mapped_original_var(decomp, sp_id, sp_var)
        (_, up), cut_info = Vertigo.BranchCutPrice.create_branching_children(
            space.id_counter, root, orig_var, 0.5, ws, root.dual_bound,
            space.cut_tracker
        )
        for (cut_id, ov) in cut_info
            space.branching_cut_info[cut_id] = ov
        end

        # A column using the branched variable once, generated while the
        # constraint is not active: its coefficient there is 1.0.
        sol = Vertigo.Reformulation._SpSolution(sp_id, -1.0, [(sp_var, 1.0)])
        col_var = Vertigo.ColGen._insert_column!(
            ws, Vertigo.ColGen.Phase2(),
            Vertigo.ColGen.PricingPrimalSolution(sp_id, sol, true)
        )

        Vertigo.TreeSearch.transition!(space, root, up)
        Vertigo.BranchCutPrice._rebuild_branching_constraints!(space)

        (bc,) = Vertigo.Branching.bp_branching_constraints(ws)
        f = Vertigo.Reformulation.with_typed_ci(bc.constraint_index) do ci
            MOI.get(
                Vertigo.Branching.bp_master_model(ws),
                MOI.ConstraintFunction(), ci
            )
        end
        coeffs = Dict(t.variable => t.coefficient for t in f.terms)
        @test !isnothing(col_var)
        @test get(coeffs, col_var, 0.0) ≈ 1.0 atol = 1e-10
    end
end
