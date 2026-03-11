# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Cover inequality separator for GAP ───────────────────────────────

function _make_cover_separator(inst::GAPInstance)
    return CustomCutSeparator(
        function (x::Dict{Tuple{Int,Int},Float64})
            cuts = SeparatedCut{Tuple{Int,Int}}[]
            for k in 1:inst.n_machines
                tasks_with_vals = Tuple{Int,Float64}[]
                for t in 1:inst.n_tasks
                    val = get(x, (k, t), 0.0)
                    if val > 1e-6
                        push!(tasks_with_vals, (t, val))
                    end
                end
                sort!(tasks_with_vals; by=tv -> -tv[2])

                cumw = 0.0
                cover = Int[]
                for (t, _) in tasks_with_vals
                    push!(cover, t)
                    cumw += inst.weight[k, t]
                    if cumw > inst.capacity[k]
                        break
                    end
                end

                if cumw <= inst.capacity[k]
                    continue
                end

                # Check violation: sum of x vals > |S| - 1
                lhs_val = sum(
                    get(x, (k, t), 0.0) for t in cover
                )
                rhs = length(cover) - 1.0
                if lhs_val > rhs + 1e-6
                    coeffs = Dict{Tuple{Int,Int},Float64}(
                        (k, t) => 1.0 for t in cover
                    )
                    push!(
                        cuts,
                        SeparatedCut{Tuple{Int,Int}}(
                            coeffs,
                            MOI.LessThan(rhs)
                        )
                    )
                end
            end
            return cuts
        end
    )
end

# ── Test: BCP with cover cuts on small GAP ───────────────────────────

function test_cut_col_gen_gap()
    @testset "[bcp+cuts] small GAP with cover inequalities" begin
        inst = gap_small_feasible2()
        ctx = build_gap_context(inst; smoothing_alpha=0.5)
        separator = _make_cover_separator(inst)

        output = run_branch_and_price(
            ctx;
            separator=separator,
            max_cut_rounds=5,
            node_limit=5_000,
            log=false
        )

        @test output.status in (:optimal, :node_limit)

        if output.status == :optimal
            @test !isnothing(output.incumbent)
            @test abs(output.incumbent.obj_value - 75.0) <= 1e-4
        end

        if !isnothing(output.incumbent)
            @test output.best_dual_bound <=
                output.incumbent.obj_value + 1e-4
        end
    end
end

# ── Test: BCP with cover cuts on benchmark GAP A-10-100 ──────────────

function test_cut_col_gen_benchmark()
    @testset "[bcp+cuts] GAP A-10-100 with cover inequalities" begin
        filepath = get_gap_instance_path('A', 10, 100)
        inst = parse_gap_file(filepath)
        ctx = build_gap_context(inst; smoothing_alpha=0.5)
        separator = _make_cover_separator(inst)

        output = run_branch_and_price(
            ctx;
            separator=separator,
            max_cut_rounds=3,
            node_limit=5_000,
            log=false
        )

        @test output.status in (:optimal, :node_limit)
        @test output.best_dual_bound <= 1360.0 + 1e-4

        if !isnothing(output.incumbent)
            @test output.best_dual_bound <=
                output.incumbent.obj_value + 1e-4
        end

        if output.status == :optimal
            @test !isnothing(output.incumbent)
            @test abs(output.incumbent.obj_value - 1360.0) <= 1e-4
        end
    end
end

# ── Wrapper ──────────────────────────────────────────────────────────

function test_cut_col_gen_e2e()
    @testset "Branch-Cut-Price with cover inequalities" begin
        test_cut_col_gen_gap()
        test_cut_col_gen_benchmark()
    end
end
