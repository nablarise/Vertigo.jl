# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Fenchel cut separation for GAP ──────────────────────────────

"""
    _enumerate_feasible_configs(Q, total_weight)

Enumerate all machine configurations h ∈ {0,1}^M such that
h'Q >= total_weight (enough capacity to serve all tasks).
"""
function _enumerate_feasible_configs(
    Q::Vector{Float64}, total_weight::Float64
)
    M = length(Q)
    configs = Vector{Int}[]
    for mask in 1:(2^M - 1)
        h = [(mask >> (m - 1)) & 1 for m in 1:M]
        if sum(h[m] * Q[m] for m in 1:M) >= total_weight
            push!(configs, h)
        end
    end
    return configs
end

"""
    _make_fenchel_separator(inst)

Build a Fenchel cut separator for a GAP instance with opening
costs. Enumerates feasible machine configurations once, then
solves a dual separation LP at each call.
"""
function _make_fenchel_separator(
    inst::GAPInstanceWithOpeningCosts
)
    M = inst.n_machines
    total_weight = sum(inst.weight)
    configs = _enumerate_feasible_configs(
        inst.capacity, total_weight
    )

    sep_model = Model(HiGHS.Optimizer)
    set_silent(sep_model)
    @variable(sep_model, ψ[1:M] >= 0)
    for h in configs
        @constraint(
            sep_model,
            sum(ψ[m] * h[m] for m in 1:M) >= 1
        )
    end

    return CustomCutSeparator(
        function (x)
            ȳ = [get(x, (:y, m, 0), 0.0) for m in 1:M]
            @objective(
                sep_model, Min,
                sum(ȳ[m] * ψ[m] for m in 1:M)
            )
            optimize!(sep_model)
            T = Tuple{Symbol,Int,Int}
            if termination_status(sep_model) != OPTIMAL
                return SeparatedCut{T}[]
            end
            obj = objective_value(sep_model)
            if obj < 1.0 - 1e-6
                coeffs = Dict{T,Float64}()
                for m in 1:M
                    v = value(ψ[m])
                    if abs(v) > 1e-10
                        coeffs[(:y, m, 0)] = v
                    end
                end
                return [SeparatedCut{T}(
                    coeffs, MOI.GreaterThan(1.0)
                )]
            end
            return SeparatedCut{T}[]
        end
    )
end

# ── Unit test for Fenchel separation ─────────────────────────────

function test_fenchel_separation()
    @testset "[fenchel] dual separation LP" begin
        # 3 machines, capacities large enough that any 2
        # suffice: Q = [10, 10, 10], total_weight = 15
        Q = [10.0, 10.0, 10.0]
        total_w = 15.0
        configs = _enumerate_feasible_configs(Q, total_w)

        sep = Model(HiGHS.Optimizer)
        set_silent(sep)
        @variable(sep, ψ[1:3] >= 0)
        for h in configs
            @constraint(
                sep,
                sum(ψ[m] * h[m] for m in 1:3) >= 1
            )
        end

        # Fractional solution: all machines half-open
        ȳ = [0.5, 0.5, 0.5]
        @objective(sep, Min, sum(ȳ[m] * ψ[m] for m in 1:3))
        optimize!(sep)

        @test termination_status(sep) == OPTIMAL
        obj = objective_value(sep)
        # Any 2 machines suffice → ψ'h >= 1 for all
        # pairs. With ȳ = 0.5 each, violated if obj < 1.
        @test obj < 1.0 - 1e-6

        ψ_vals = [value(ψ[m]) for m in 1:3]
        @test all(v -> v >= -1e-10, ψ_vals)
    end
end

# ── E2e test: BCP with Fenchel cuts on GAP ──────────────────────

function test_fenchel_cut_gap()
    @testset "[bcp+cuts] GAP with Fenchel cuts" begin
        inst = gap_fenchel_instance()
        ctx = build_gap_fenchel_context(inst)
        separator = _make_fenchel_separator(inst)

        output = run_branch_and_price(
            ctx;
            separator=separator,
            max_cut_rounds=20,
            min_gap_improvement=0.0,
            node_limit=1,
            log_level=2
        )

        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
        @test abs(
            output.incumbent.obj_value - 440.55
        ) <= 1e-2
    end
end

# ── Wrapper ──────────────────────────────────────────────────────

function test_cut_col_gen_e2e()
    @testset "Branch-Cut-Price with Fenchel cuts" begin
        test_fenchel_separation()
        test_fenchel_cut_gap()
    end
end
