# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct GAPInstance
    n_machines::Int
    n_tasks::Int
    cost::Matrix{Float64}      # cost[k,t]
    weight::Matrix{Float64}    # weight[k,t]
    capacity::Vector{Float64}  # capacity[k]
end

function random_gap_instance(n_machines, n_tasks; seed=42)
    rng = Random.MersenneTwister(seed)
    cost = rand(rng, 1:20, n_machines, n_tasks) .* 1.0
    weight = rand(rng, 1:5, n_machines, n_tasks) .* 1.0
    capacity = [ceil(0.7 * sum(weight[k, :])) for k in 1:n_machines]
    return GAPInstance(n_machines, n_tasks, cost, weight, capacity)
end

"""
    parse_gap_file(filepath::String) -> GAPInstance

Parse a GAP benchmark instance file into a `GAPInstance`.

Numbers are read as a flat whitespace-separated token stream regardless of
how they are laid out across lines (classes C/D use ~12 per line; class E
packs all jobs for one machine on a single line).
"""
function parse_gap_file(filepath::String)::GAPInstance
    tokens = split(read(filepath, String))
    idx = Ref(1)

    next_int() = (v = parse(Int, tokens[idx[]]); idx[] += 1; v)

    m = next_int()
    n = next_int()

    cost = Matrix{Float64}(undef, m, n)
    for i in 1:m, j in 1:n
        cost[i, j] = next_int()
    end

    weight = Matrix{Float64}(undef, m, n)
    for i in 1:m, j in 1:n
        weight[i, j] = next_int()
    end

    capacity = Vector{Float64}(undef, m)
    for i in 1:m
        capacity[i] = next_int()
    end

    return GAPInstance(m, n, cost, weight, capacity)
end

"""
    get_gap_instance_path(class::Char, agents::Int, jobs::Int) -> String

Extract a named GAP instance from its zip archive and return the path to the
extracted file in a temporary directory.

The zip archives are located at `test/data/gap/gap_{c,d,e}.zip` relative to
this file's directory. The entry name follows the convention `{class}{agents:02d}{jobs}`,
e.g. `('C', 5, 100)` → `"c05100"`.
"""
function get_gap_instance_path(class::Char, agents::Int, jobs::Int)::String
    lc = lowercase(class)
    zip_path = joinpath(@__DIR__, "..", "data", "gap", "gap_$(lc).zip")
    entry_name = "$(lc)$(lpad(agents, 2, '0'))$(jobs)"

    out_path = joinpath(mktempdir(), entry_name)

    zf = ZipFile.Reader(zip_path)
    found = false
    for f in zf.files
        if f.name == entry_name
            write(out_path, read(f))
            found = true
            break
        end
    end
    close(zf)

    found || error("Entry '$(entry_name)' not found in '$(zip_path)'")
    return out_path
end

# ──────────────────────────────────────────────────────────────────────
# Build ColGenContext for a GAP instance
# ──────────────────────────────────────────────────────────────────────

function build_gap_context(inst::GAPInstance; smoothing_alpha::Float64=0.0)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    # ── Master model ──────────────────────────────────────────────────
    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)

    @constraint(master_jump, assignment[t in T], 0 == 1)   # coupling: = 1
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)      # Σλ ≥ 0
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)      # Σλ ≤ 1
    @objective(master_jump, Min, 0)

    master_model = backend(master_jump)

    # ── Subproblem models ─────────────────────────────────────────────
    sp_models = Dict{PricingSubproblemId,Any}()
    sp_var_indices = Dict{PricingSubproblemId,Vector{MOI.VariableIndex}}()

    for k in K
        sp_jump = Model(HiGHS.Optimizer)
        set_silent(sp_jump)

        @variable(sp_jump, z[t in T], Bin)
        @constraint(sp_jump, sum(inst.weight[k, t] * z[t] for t in T) <= inst.capacity[k])
        @objective(sp_jump, Min, sum(inst.cost[k, t] * z[t] for t in T))

        sp_models[PricingSubproblemId(k)] = backend(sp_jump)
        sp_var_indices[PricingSubproblemId(k)] = [index(z[t]) for t in T]
    end

    # ── Build Decomposition ───────────────────────────────────────────
    SpVar = MOI.VariableIndex
    CstrId = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}

    builder = DecompositionBuilder{Tuple{Int,Int}}(minimize=true)

    for k in K
        add_subproblem!(builder, PricingSubproblemId(k), 0.0, 0.0, 1.0)
    end

    for k in K
        for t in T
            sp_var = sp_var_indices[PricingSubproblemId(k)][t]
            add_sp_variable!(builder, PricingSubproblemId(k), sp_var, inst.cost[k, t])
            cstr_idx = index(assignment[t])
            add_coupling_coefficient!(builder, PricingSubproblemId(k), sp_var, cstr_idx, 1.0)
            add_mapping!(builder, (k, t), PricingSubproblemId(k), sp_var)
        end
    end

    for t in T
        add_coupling_constraint!(builder, index(assignment[t]), 1.0)
    end

    decomp = build(builder)

    # ── Column pool ───────────────────────────────────────────────────
    pool = ColumnPool()

    # ── Convexity constraint indices ──────────────────────────────────
    conv_ub_map = Dict{PricingSubproblemId,Any}(PricingSubproblemId(k) => index(conv_ub[k]) for k in K)
    conv_lb_map = Dict{PricingSubproblemId,Any}(PricingSubproblemId(k) => index(conv_lb[k]) for k in K)

    # ── Build context ─────────────────────────────────────────────────
    inner_ctx = ColGenContext(
        decomp,
        master_model,
        conv_ub_map,
        conv_lb_map,
        sp_models,
        pool,
        NonRobustCutManager{CstrId}(),
        Dict{Any,Any}(),
        Dict{Any,Any}(),
        Dict{Any,Any}();
        smoothing_alpha=smoothing_alpha
    )
    ctx = ColGenLoggerContext(inner_ctx)

    return ctx
end

function build_gap_context_with_fixed_cost(
    inst::GAPInstance, fixed_cost::Float64
)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    # ── Master model ──────────────────────────────────────────────────
    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)

    @constraint(master_jump, assignment[t in T], 0 == 1)
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)
    @objective(master_jump, Min, fixed_cost)

    master_model = backend(master_jump)

    # ── Subproblem models ─────────────────────────────────────────────
    sp_models = Dict{PricingSubproblemId,Any}()
    sp_var_indices = Dict{PricingSubproblemId,Vector{MOI.VariableIndex}}()

    for k in K
        sp_jump = Model(HiGHS.Optimizer)
        set_silent(sp_jump)

        @variable(sp_jump, z[t in T], Bin)
        @constraint(
            sp_jump,
            sum(inst.weight[k, t] * z[t] for t in T) <= inst.capacity[k]
        )
        @objective(sp_jump, Min, sum(inst.cost[k, t] * z[t] for t in T))

        sp_models[PricingSubproblemId(k)] = backend(sp_jump)
        sp_var_indices[PricingSubproblemId(k)] = [index(z[t]) for t in T]
    end

    # ── Build Decomposition ───────────────────────────────────────────
    CstrId = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}
    }

    builder = DecompositionBuilder{Tuple{Int,Int}}(minimize=true)

    for k in K
        add_subproblem!(builder, PricingSubproblemId(k), 0.0, 0.0, 1.0)
    end

    for k in K
        for t in T
            sp_var = sp_var_indices[PricingSubproblemId(k)][t]
            add_sp_variable!(
                builder, PricingSubproblemId(k), sp_var, inst.cost[k, t]
            )
            cstr_idx = index(assignment[t])
            add_coupling_coefficient!(
                builder, PricingSubproblemId(k), sp_var, cstr_idx, 1.0
            )
            add_mapping!(builder, (k, t), PricingSubproblemId(k), sp_var)
        end
    end

    for t in T
        add_coupling_constraint!(builder, index(assignment[t]), 1.0)
    end

    decomp = build(builder)

    # ── Column pool ───────────────────────────────────────────────────
    pool = ColumnPool()

    # ── Convexity constraint indices ──────────────────────────────────
    conv_ub_map = Dict{PricingSubproblemId,Any}(
        PricingSubproblemId(k) => index(conv_ub[k]) for k in K
    )
    conv_lb_map = Dict{PricingSubproblemId,Any}(
        PricingSubproblemId(k) => index(conv_lb[k]) for k in K
    )

    # ── Build context ─────────────────────────────────────────────────
    inner_ctx = ColGenContext(
        decomp,
        master_model,
        conv_ub_map,
        conv_lb_map,
        sp_models,
        pool,
        NonRobustCutManager{CstrId}(),
        Dict{Any,Any}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    ctx = ColGenLoggerContext(inner_ctx)

    return ctx
end

# ──────────────────────────────────────────────────────────────────────
# Build ColGenContext for a shifted GAP (z ∈ {1,2} instead of x ∈ {0,1})
# ──────────────────────────────────────────────────────────────────────

function build_gap_shifted_context(inst::GAPInstance)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    # ── Master model ──────────────────────────────────────────────────
    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)

    @constraint(master_jump, assignment[t in T], 0 >= 1)   # coupling: ≥ 1
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)      # Σλ ≥ 0
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)      # Σλ ≤ 1
    @objective(master_jump, Min, 0)

    master_model = backend(master_jump)

    # ── Subproblem models ─────────────────────────────────────────────
    sp_models = Dict{PricingSubproblemId,Any}()
    sp_var_indices = Dict{PricingSubproblemId,Vector{MOI.VariableIndex}}()
    sp_one_indices = Dict{PricingSubproblemId,MOI.VariableIndex}()

    for k in K
        sp_jump = Model(HiGHS.Optimizer)
        set_silent(sp_jump)

        @variable(sp_jump, 1 <= z[t in T] <= 2, Int)
        @variable(sp_jump, one == 1)
        shifted_cap = sum(inst.weight[k, :]) + inst.capacity[k]
        @constraint(sp_jump, sum(inst.weight[k, t] * z[t] for t in T) <= shifted_cap)
        @objective(sp_jump, Min, sum(inst.cost[k, t] * z[t] for t in T))

        sp_models[PricingSubproblemId(k)] = backend(sp_jump)
        sp_var_indices[PricingSubproblemId(k)] = [index(z[t]) for t in T]
        sp_one_indices[PricingSubproblemId(k)] = index(one)
    end

    # ── Build Decomposition ───────────────────────────────────────────
    CstrId = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}

    builder = DecompositionBuilder{Tuple{Int,Int}}(minimize=true)

    for k in K
        fixed_cost = -sum(inst.cost[k, :])
        add_subproblem!(builder, PricingSubproblemId(k), fixed_cost, 0.0, 1.0)
    end

    for k in K
        sp_id = PricingSubproblemId(k)
        for t in T
            sp_var = sp_var_indices[sp_id][t]
            add_sp_variable!(builder, sp_id, sp_var, inst.cost[k, t])
            cstr_idx = index(assignment[t])
            add_coupling_coefficient!(builder, sp_id, sp_var, cstr_idx, 1.0)
            add_mapping!(builder, (k, t), sp_id, sp_var)
        end

        # Fixed variable `one == 1` with coupling coefficient -1.0
        one_var = sp_one_indices[sp_id]
        add_sp_variable!(builder, sp_id, one_var, 0.0)
        for t in T
            cstr_idx = index(assignment[t])
            add_coupling_coefficient!(builder, sp_id, one_var, cstr_idx, -1.0)
        end
    end

    for t in T
        add_coupling_constraint!(builder, index(assignment[t]), 1.0)
    end

    decomp = build(builder)

    # ── Column pool ───────────────────────────────────────────────────
    pool = ColumnPool()

    # ── Convexity constraint indices ──────────────────────────────────
    conv_ub_map = Dict{PricingSubproblemId,Any}(PricingSubproblemId(k) => index(conv_ub[k]) for k in K)
    conv_lb_map = Dict{PricingSubproblemId,Any}(PricingSubproblemId(k) => index(conv_lb[k]) for k in K)

    # ── Build context ─────────────────────────────────────────────────
    inner_ctx = ColGenContext(
        decomp,
        master_model,
        conv_ub_map,
        conv_lb_map,
        sp_models,
        pool,
        NonRobustCutManager{CstrId}(),
        Dict{Any,Any}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    ctx = ColGenLoggerContext(inner_ctx)

    return ctx
end

struct GAPInstanceWithIdenticalMachines
    n_machine_types::Int
    n_tasks::Int
    cost::Matrix{Float64}      # cost[type, task]
    weight::Matrix{Float64}    # weight[type, task]
    capacity::Vector{Float64}  # capacity[type]
    multiplicity::Vector{Int}  # multiplicity[type]
end

function build_gap_identical_context(
    inst::GAPInstanceWithIdenticalMachines
)
    K = 1:inst.n_machine_types
    T = 1:inst.n_tasks

    # ── Master model ──────────────────────────────────────────────────
    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)

    @constraint(master_jump, assignment[t in T], 0 == 1)
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)
    @constraint(
        master_jump,
        conv_ub[k in K],
        0 <= inst.multiplicity[k]
    )
    @objective(master_jump, Min, 0)

    master_model = backend(master_jump)

    # ── Subproblem models ─────────────────────────────────────────────
    sp_models = Dict{PricingSubproblemId,Any}()
    sp_var_indices = Dict{
        PricingSubproblemId,Vector{MOI.VariableIndex}
    }()

    for k in K
        sp_jump = Model(HiGHS.Optimizer)
        set_silent(sp_jump)

        @variable(sp_jump, z[t in T], Bin)
        @constraint(
            sp_jump,
            sum(
                inst.weight[k, t] * z[t] for t in T
            ) <= inst.capacity[k]
        )
        @objective(
            sp_jump,
            Min,
            sum(inst.cost[k, t] * z[t] for t in T)
        )

        sp_models[PricingSubproblemId(k)] = backend(sp_jump)
        sp_var_indices[PricingSubproblemId(k)] = [
            index(z[t]) for t in T
        ]
    end

    # ── Build Decomposition ───────────────────────────────────────────
    CstrId = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.EqualTo{Float64}
    }

    builder = DecompositionBuilder{Tuple{Int,Int}}(
        minimize=true
    )

    for k in K
        add_subproblem!(
            builder,
            PricingSubproblemId(k),
            0.0,
            0.0,
            Float64(inst.multiplicity[k])
        )
    end

    for k in K
        for t in T
            sp_var = sp_var_indices[PricingSubproblemId(k)][t]
            add_sp_variable!(
                builder,
                PricingSubproblemId(k),
                sp_var,
                inst.cost[k, t]
            )
            cstr_idx = index(assignment[t])
            add_coupling_coefficient!(
                builder,
                PricingSubproblemId(k),
                sp_var,
                cstr_idx,
                1.0
            )
            add_mapping!(
                builder,
                (k, t),
                PricingSubproblemId(k),
                sp_var
            )
        end
    end

    for t in T
        add_coupling_constraint!(
            builder, index(assignment[t]), 1.0
        )
    end

    decomp = build(builder)

    # ── Column pool ───────────────────────────────────────────────────
    pool = ColumnPool()

    # ── Convexity constraint indices ──────────────────────────────────
    conv_ub_map = Dict{PricingSubproblemId,Any}(
        PricingSubproblemId(k) => index(conv_ub[k]) for k in K
    )
    conv_lb_map = Dict{PricingSubproblemId,Any}(
        PricingSubproblemId(k) => index(conv_lb[k]) for k in K
    )

    # ── Build context ─────────────────────────────────────────────────
    inner_ctx = ColGenContext(
        decomp,
        master_model,
        conv_ub_map,
        conv_lb_map,
        sp_models,
        pool,
        NonRobustCutManager{CstrId}(),
        Dict{Any,Any}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    ctx = ColGenLoggerContext(inner_ctx)

    return ctx
end

struct GAPWithPenaltyInstance
    gap::GAPInstance
    penalty::Vector{Float64}   # penalty[t] for leaving task t unassigned
end

function build_gap_with_penalty_context(inst::GAPWithPenaltyInstance)
    gap = inst.gap
    K = 1:gap.n_machines
    T = 1:gap.n_tasks

    # ── Master model ──────────────────────────────────────────────────
    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)

    @variable(master_jump, u[t in T], Bin)
    @constraint(master_jump, assignment[t in T], 0 == 1)   # coupling: = 1
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)      # Σλ ≥ 0
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)      # Σλ ≤ 1
    @objective(master_jump, Min, sum(inst.penalty[t] * u[t] for t in T))

    for t in T
        set_normalized_coefficient(assignment[t], u[t], 1.0)
    end

    master_model = backend(master_jump)

    # ── Subproblem models ─────────────────────────────────────────────
    sp_models = Dict{PricingSubproblemId,Any}()
    sp_var_indices = Dict{PricingSubproblemId,Vector{MOI.VariableIndex}}()

    for k in K
        sp_jump = Model(HiGHS.Optimizer)
        set_silent(sp_jump)

        @variable(sp_jump, z[t in T], Bin)
        @constraint(sp_jump, sum(gap.weight[k, t] * z[t] for t in T) <= gap.capacity[k])
        @objective(sp_jump, Min, sum(gap.cost[k, t] * z[t] for t in T))

        sp_models[PricingSubproblemId(k)] = backend(sp_jump)
        sp_var_indices[PricingSubproblemId(k)] = [index(z[t]) for t in T]
    end

    # ── Build Decomposition ───────────────────────────────────────────
    CstrId = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}

    builder = DecompositionBuilder{Tuple{Int,Int}}(minimize=true)

    for k in K
        add_subproblem!(builder, PricingSubproblemId(k), 0.0, 0.0, 1.0)
    end

    for k in K
        for t in T
            sp_var = sp_var_indices[PricingSubproblemId(k)][t]
            add_sp_variable!(builder, PricingSubproblemId(k), sp_var, gap.cost[k, t])
            cstr_idx = index(assignment[t])
            add_coupling_coefficient!(builder, PricingSubproblemId(k), sp_var, cstr_idx, 1.0)
            add_mapping!(builder, (k, t), PricingSubproblemId(k), sp_var)
        end
    end

    for t in T
        add_coupling_constraint!(builder, index(assignment[t]), 1.0)
        add_pure_master_variable!(builder, index(u[t]), inst.penalty[t], 0.0, 1.0, true)
        add_pure_master_coupling!(builder, index(u[t]), index(assignment[t]), 1.0)
    end

    decomp = build(builder)

    # ── Column pool ───────────────────────────────────────────────────
    pool = ColumnPool()

    # ── Convexity constraint indices ──────────────────────────────────
    conv_ub_map = Dict{PricingSubproblemId,Any}(PricingSubproblemId(k) => index(conv_ub[k]) for k in K)
    conv_lb_map = Dict{PricingSubproblemId,Any}(PricingSubproblemId(k) => index(conv_lb[k]) for k in K)

    # ── Build context ─────────────────────────────────────────────────
    inner_ctx = ColGenContext(
        decomp,
        master_model,
        conv_ub_map,
        conv_lb_map,
        sp_models,
        pool,
        NonRobustCutManager{CstrId}(),
        Dict{Any,Any}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    ctx = ColGenLoggerContext(inner_ctx)

    return ctx
end

function build_gap_context_max(inst::GAPInstance)
    K = 1:inst.n_machines
    T = 1:inst.n_tasks

    # ── Master model ──────────────────────────────────────────────────
    master_jump = Model(HiGHS.Optimizer)
    set_silent(master_jump)

    @constraint(master_jump, assignment[t in T], 0 == 1)   # coupling: = 1
    @constraint(master_jump, conv_lb[k in K], 0 >= 0)      # Σλ ≥ 0
    @constraint(master_jump, conv_ub[k in K], 0 <= 1)      # Σλ ≤ 1
    @objective(master_jump, Max, 0)                          # maximization sense

    master_model = backend(master_jump)

    # ── Subproblem models ─────────────────────────────────────────────
    sp_models = Dict{PricingSubproblemId,Any}()
    sp_var_indices = Dict{PricingSubproblemId,Vector{MOI.VariableIndex}}()

    for k in K
        sp_jump = Model(HiGHS.Optimizer)
        set_silent(sp_jump)

        @variable(sp_jump, z[t in T], Bin)
        @constraint(sp_jump, sum(inst.weight[k, t] * z[t] for t in T) <= inst.capacity[k])
        @objective(sp_jump, Min, sum(inst.cost[k, t] * z[t] for t in T))

        sp_models[PricingSubproblemId(k)] = backend(sp_jump)
        sp_var_indices[PricingSubproblemId(k)] = [index(z[t]) for t in T]
    end

    # ── Build Decomposition ───────────────────────────────────────────
    SpVar = MOI.VariableIndex
    CstrId = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}

    builder = DecompositionBuilder{Tuple{Int,Int}}(minimize=false)

    for k in K
        add_subproblem!(builder, PricingSubproblemId(k), 0.0, 0.0, 1.0)
    end

    for k in K
        for t in T
            sp_var = sp_var_indices[PricingSubproblemId(k)][t]
            add_sp_variable!(builder, PricingSubproblemId(k), sp_var, inst.cost[k, t])
            cstr_idx = index(assignment[t])
            add_coupling_coefficient!(builder, PricingSubproblemId(k), sp_var, cstr_idx, 1.0)
            add_mapping!(builder, (k, t), PricingSubproblemId(k), sp_var)
        end
    end

    for t in T
        add_coupling_constraint!(builder, index(assignment[t]), 1.0)
    end

    decomp = build(builder)

    # ── Column pool ───────────────────────────────────────────────────
    pool = ColumnPool()

    # ── Convexity constraint indices ──────────────────────────────────
    conv_ub_map = Dict{PricingSubproblemId,Any}(PricingSubproblemId(k) => index(conv_ub[k]) for k in K)
    conv_lb_map = Dict{PricingSubproblemId,Any}(PricingSubproblemId(k) => index(conv_lb[k]) for k in K)

    # ── Build context ─────────────────────────────────────────────────
    inner_ctx = ColGenContext(
        decomp,
        master_model,
        conv_ub_map,
        conv_lb_map,
        sp_models,
        pool,
        NonRobustCutManager{CstrId}(),
        Dict{Any,Any}(),
        Dict{Any,Any}(),
        Dict{Any,Any}()
    )
    ctx = ColGenLoggerContext(inner_ctx)

    return ctx
end
