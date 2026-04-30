# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# ── Node data ────────────────────────────────────────────────────────────

"""
    BPNodeData

User data attached to each search node. Stores the column generation
output after evaluation, plus branching context set by `branch!`.
"""
mutable struct BPNodeData
    cg_output::Union{Nothing,ColGen.ColGenOutput}
    branching_var::Any
    parent_lp_obj::Union{Nothing,Float64}
    branching_direction::Union{Nothing,BranchingDirection}
    branching_frac::Union{Nothing,Float64}
end

BPNodeData() = BPNodeData(nothing, nothing, nothing, nothing, nothing)

# ── Search space ─────────────────────────────────────────────────────────

"""
    BPSpace <: TreeSearch.AbstractSearchSpace

Search space for branch-and-price. Wraps the CG workspace, the MOI
backend, domain/cut tracking, and branching metadata.
"""
mutable struct BPSpace{Ws,B,S<:Union{Nothing,AbstractCutSeparator}} <: TreeSearch.AbstractSearchSpace
    ws::Ws
    backend::B
    domain_helper::MathOptState.DomainChangeTrackerHelper
    cut_tracker::MathOptState.LocalCutTracker
    cut_helper::MathOptState.LocalCutTrackerHelper
    branching_cut_info::Dict{Int,Any}
    id_counter::TreeSearch.NodeIdCounter
    incumbent::Union{Nothing,ColGen.MasterIpPrimalSol}
    last_ip_incumbent::Union{Nothing,ColGen.MasterIpPrimalSol}
    best_dual_bound::Float64
    open_node_bounds::Dict{Int,Float64}
    nodes_explored::Int
    node_limit::Int
    tol::Float64
    rmp_time_limit::Float64
    rmp_heuristic::Bool
    separator::S
    cutcolgen_ctx::CutColGenContext
    total_cuts_separated::Int
    branching_strategy::AbstractBranchingStrategy
    log_level::Int
end

"""
    BPSpace(ws; node_limit=10_000, tol=1e-6, rmp_time_limit=60.0,
            rmp_heuristic=true, separator=nothing,
            max_cut_rounds=0, min_gap_improvement=0.01)

Create a branch-and-price search space from a column generation
workspace. Registers existing variable bound constraints for tracking.
"""
function BPSpace(
    ws::Union{ColGen.ColGenWorkspace,ColGen.ColGenLoggerWorkspace};
    node_limit::Int = 10_000,
    tol::Float64 = 1e-6,
    rmp_time_limit::Float64 = 60.0,
    rmp_heuristic::Bool = true,
    separator::Union{Nothing,AbstractCutSeparator} = nothing,
    max_cut_rounds::Int = 0,
    min_gap_improvement::Float64 = 0.01,
    branching_strategy::AbstractBranchingStrategy = MostFractionalBranching(),
    log_level::Int = 0
)
    master = bp_master_model(ws)
    tracker = MathOptState.DomainChangeTracker()
    domain_helper = MathOptState.transform_model!(
        tracker, master
    )
    cut_tracker = MathOptState.LocalCutTracker()
    cut_helper = MathOptState.transform_model!(
        cut_tracker, master
    )
    return BPSpace(
        ws, master, domain_helper,
        cut_tracker, cut_helper,
        Dict{Int,Any}(),
        TreeSearch.NodeIdCounter(),
        nothing, nothing,
        is_minimization(ws) ? -Inf : Inf,
        Dict{Int,Float64}(),
        0, node_limit, tol, rmp_time_limit,
        rmp_heuristic, separator,
        CutColGenContext(max_cut_rounds, min_gap_improvement),
        0,
        branching_strategy,
        log_level
    )
end

# ── TreeSearch interface ─────────────────────────────────────────────────

function TreeSearch.new_root(space::BPSpace)
    empty_fwd = (
        MathOptState.DomainChangeDiff(),
        MathOptState.LocalCutChangeDiff()
    )
    empty_bwd = (
        MathOptState.DomainChangeDiff(),
        MathOptState.LocalCutChangeDiff()
    )
    node = TreeSearch.root_node(
        space.id_counter,
        empty_fwd,
        empty_bwd,
        BPNodeData()
    )
    space.open_node_bounds[node.id] = node.dual_bound
    return node
end

function TreeSearch.stop(space::BPSpace, _)
    return space.nodes_explored >= space.node_limit
end

function TreeSearch.output(space::BPSpace)
    status = if !isnothing(space.incumbent)
        db = space.best_dual_bound
        ub = space.incumbent.obj_value
        abs(ub - db) < space.tol ? :optimal : :node_limit
    elseif space.nodes_explored >= space.node_limit
        :node_limit
    else
        :infeasible
    end
    return BPOutput(
        status, space.incumbent,
        space.best_dual_bound, space.nodes_explored
    )
end

function TreeSearch.transition!(space::BPSpace, current, next)
    fwd!, bwd! = MathOptState.make_transition_callbacks(
        space.backend, (space.domain_helper, space.cut_helper)
    )
    TreeSearch.transition_to!(current, next, fwd!, bwd!)
    return
end

function _recompute_global_dual_bound!(space::BPSpace)
    bounds = space.open_node_bounds
    if isempty(bounds)
        space.best_dual_bound = is_minimization(space.ws) ?
            -Inf : Inf
        return
    end
    space.best_dual_bound = is_minimization(space.ws) ?
        minimum(values(bounds)) : maximum(values(bounds))
    return
end

function TreeSearch.branch!(space::BPSpace, node)
    primal_values = get_primal_solution(space.backend)

    result = select_branching_variable(
        space.branching_strategy, space, node, primal_values
    )
    if result.status == all_integral
        return typeof(node)[]
    elseif result.status == node_infeasible
        return typeof(node)[]
    end
    orig_var = result.orig_var
    x_val = result.value

    cg_output = node.user_data.cg_output
    db = if isnothing(cg_output) ||
            isnothing(cg_output.incumbent_dual_bound)
        node.dual_bound
    else
        cg_output.incumbent_dual_bound
    end

    parent_lp = if !isnothing(node.user_data) &&
                   !isnothing(node.user_data.cg_output)
        node.user_data.cg_output.master_lp_obj
    else
        nothing
    end
    branching_frac = x_val - floor(x_val)

    children, cut_info = create_branching_children(
        space.id_counter, node, orig_var, x_val,
        space.ws, db, space.cut_tracker
    )
    for (cut_id, ov) in cut_info
        space.branching_cut_info[cut_id] = ov
    end
    children[1].user_data.branching_var = orig_var
    children[1].user_data.parent_lp_obj = parent_lp
    children[1].user_data.branching_direction = branch_down
    children[1].user_data.branching_frac = branching_frac
    children[2].user_data.branching_var = orig_var
    children[2].user_data.parent_lp_obj = parent_lp
    children[2].user_data.branching_direction = branch_up
    children[2].user_data.branching_frac = branching_frac
    for child in children
        space.open_node_bounds[child.id] = child.dual_bound
    end
    _recompute_global_dual_bound!(space)
    return children
end

function TreeSearch.on_feasible_solution!(
    ::BPSpace, ::TreeSearch.SearchNode
)
    # Incumbent already updated in evaluate!
    return
end

# ── TreeSearch logger protocol ───────────────────────────────────────────

function TreeSearch.ts_incumbent_value(s::BPSpace)
    return isnothing(s.incumbent) ? nothing : s.incumbent.obj_value
end

function TreeSearch.ts_best_dual_bound(s::BPSpace)
    return s.best_dual_bound
end

function TreeSearch.ts_is_minimization(s::BPSpace)
    return is_minimization(s.ws)
end

function TreeSearch.ts_nodes_explored(s::BPSpace)
    return s.nodes_explored
end

function TreeSearch.ts_search_status_message(s::BPSpace)
    inc = s.incumbent
    if !isnothing(inc)
        db = s.best_dual_bound
        if abs(inc.obj_value - db) < s.tol
            return "Optimal solution found."
        end
    end
    if s.nodes_explored >= s.node_limit
        return "Node limit reached."
    end
    return isnothing(inc) ? "No feasible solution found." :
        "Search complete."
end

function TreeSearch.ts_open_node_count(s::BPSpace)
    return length(s.open_node_bounds)
end

function TreeSearch.ts_total_columns(s::BPSpace)
    return length(bp_pool(s.ws).by_column_var)
end

function TreeSearch.ts_active_columns(s::BPSpace)
    pool = bp_pool(s.ws)
    master = s.backend
    count = 0
    for cv in keys(pool.by_column_var)
        if MOI.is_valid(master, cv)
            count += 1
        end
    end
    return count
end

function TreeSearch.ts_total_cuts(s::BPSpace)
    return s.total_cuts_separated
end

function TreeSearch.ts_branching_description(
    s::BPSpace, node::TreeSearch.SearchNode
)
    TreeSearch.is_root(node) && return nothing
    _, cut_diff = node.local_forward_diff
    isempty(cut_diff.add_cuts) && return nothing
    cut = cut_diff.add_cuts[1].cut
    orig_var = get(s.branching_cut_info, cut.id, nothing)
    isnothing(orig_var) && return nothing
    if cut.set isa MOI.LessThan
        rhs = cut.set.upper
        return "$(orig_var) <= $(Int(rhs))"
    elseif cut.set isa MOI.GreaterThan
        rhs = cut.set.lower
        return "$(orig_var) >= $(Int(rhs))"
    end
    return nothing
end

# ── Branch-and-price workspace ───────────────────────────────────────────

"""
    BranchCutPriceWorkspace

Bundle of all state needed to run branch-and-price: a `BPSpace` and the
`BranchCutPriceConfig` it was built from. Build it once with the keyword
constructor — wiring of the branching logger context (including
pseudocost-tracker preservation) is handled there — then hand it to
[`run_branch_and_price`](@ref).
"""
struct BranchCutPriceWorkspace{Sp<:BPSpace}
    space::Sp
    config::BranchCutPriceConfig
end

"""
    BranchCutPriceWorkspace(ws, config::BranchCutPriceConfig)

Build a `BranchCutPriceWorkspace` from a column generation workspace and
a `BranchCutPriceConfig`. See [`BranchCutPriceConfig`](@ref) for the
list of user parameters.
"""
function BranchCutPriceWorkspace(
    ws::Union{ColGen.ColGenWorkspace,ColGen.ColGenLoggerWorkspace},
    config::BranchCutPriceConfig
)
    branching_ctx = config.log_level > 0 ?
        BranchingLoggerContext(; log_level=config.log_level) :
        DefaultBranchingContext()
    effective_strategy = _branching_strategy_with_ctx(
        config.branching_strategy, branching_ctx
    )
    space = BPSpace(
        ws;
        node_limit = config.node_limit,
        tol = config.tol,
        rmp_time_limit = config.rmp_time_limit,
        rmp_heuristic = config.rmp_heuristic,
        separator = config.separator,
        max_cut_rounds = config.max_cut_rounds,
        min_gap_improvement = config.min_gap_improvement,
        branching_strategy = effective_strategy,
        log_level = config.log_level
    )
    return BranchCutPriceWorkspace(space, config)
end

"""
    BranchCutPriceWorkspace(ws; kwargs...)

Convenience constructor that forwards `kwargs` to
[`BranchCutPriceConfig`](@ref) and builds the workspace.
"""
function BranchCutPriceWorkspace(
    ws::Union{ColGen.ColGenWorkspace,ColGen.ColGenLoggerWorkspace};
    kwargs...
)
    return BranchCutPriceWorkspace(ws, BranchCutPriceConfig(; kwargs...))
end

# Default: leave the strategy as-is. Specialized below for strategies
# that hold a `BranchingContext`.
_branching_strategy_with_ctx(s::AbstractBranchingStrategy, _) = s

function _branching_strategy_with_ctx(
    s::MultiPhaseStrongBranching, ctx::BranchingContext
)
    return with_branching_ctx(s, ctx)
end

# ── Entry point ──────────────────────────────────────────────────────────

"""
    run_branch_and_price(bcp_ws::BranchCutPriceWorkspace) -> BPOutput

Run the branch-and-price algorithm described by `bcp_ws`.
"""
function run_branch_and_price(bcp_ws::BranchCutPriceWorkspace)
    space = bcp_ws.space
    strategy = bcp_ws.config.strategy
    evaluator = BPEvaluator()
    if !isnothing(bcp_ws.config.dot_file)
        dot_ctx = BPDotLoggerContext(
            space, evaluator, bcp_ws.config.dot_file
        )
        return TreeSearch.search(strategy, dot_ctx)
    end
    if space.log_level > 0
        ts_ctx = TreeSearch.TreeSearchLoggerContext(
            space, evaluator, space.log_level
        )
        return TreeSearch.search(strategy, ts_ctx)
    end
    return TreeSearch.search(strategy, space, evaluator)
end
