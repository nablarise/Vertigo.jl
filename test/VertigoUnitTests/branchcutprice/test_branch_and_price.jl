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

    @testset "[branch_and_price] the dot node label shows the node bound" begin
        io = IOBuffer()
        Vertigo.BranchCutPrice._write_dot_node(
            io, (id = 3, dual_bound = 410.5),
            Vertigo.BranchCutPrice.DotBranched, nothing
        )
        @test occursin("DB = 410.50", String(take!(io)))
    end
end
