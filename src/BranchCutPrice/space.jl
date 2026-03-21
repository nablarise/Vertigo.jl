# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Node data ────────────────────────────────────────────────────────────

"""
    BPNodeData

User data attached to each search node. Stores the column generation
output after evaluation.
"""
mutable struct BPNodeData
    cg_output::Union{Nothing,ColGen.ColGenOutput}
end

BPNodeData() = BPNodeData(nothing)

# ── Search space ─────────────────────────────────────────────────────────

"""
    BPSpace <: TreeSearch.AbstractSearchSpace

Search space for branch-and-price. Wraps the CG context, the MOI
backend, domain/cut tracking, and branching metadata.
"""
mutable struct BPSpace{Ctx,B,S<:Union{Nothing,AbstractCutSeparator}} <: TreeSearch.AbstractSearchSpace
    ctx::Ctx
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
end

"""
    BPSpace(ctx; node_limit=10_000, tol=1e-6, rmp_time_limit=60.0,
            rmp_heuristic=true, separator=nothing,
            max_cut_rounds=0, min_gap_improvement=0.01)

Create a branch-and-price search space from a column generation
context. Registers existing variable bound constraints for tracking.
"""
function BPSpace(
    ctx::Union{ColGen.ColGenContext,ColGen.ColGenLoggerContext};
    node_limit::Int = 10_000,
    tol::Float64 = 1e-6,
    rmp_time_limit::Float64 = 60.0,
    rmp_heuristic::Bool = true,
    separator::Union{Nothing,AbstractCutSeparator} = nothing,
    max_cut_rounds::Int = 0,
    min_gap_improvement::Float64 = 0.01,
    branching_strategy::AbstractBranchingStrategy = MostFractionalBranching()
)
    master = bp_master_model(ctx)
    tracker = MathOptState.DomainChangeTracker()
    domain_helper = MathOptState.transform_model!(
        tracker, master
    )
    cut_tracker = MathOptState.LocalCutTracker()
    cut_helper = MathOptState.transform_model!(
        cut_tracker, master
    )
    return BPSpace(
        ctx, master, domain_helper,
        cut_tracker, cut_helper,
        Dict{Int,Any}(),
        TreeSearch.NodeIdCounter(),
        nothing, nothing,
        is_minimization(ctx) ? -Inf : Inf,
        Dict{Int,Float64}(),
        0, node_limit, tol, rmp_time_limit,
        rmp_heuristic, separator,
        CutColGenContext(max_cut_rounds, min_gap_improvement),
        0,
        branching_strategy
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
        space.best_dual_bound = is_minimization(space.ctx) ?
            -Inf : Inf
        return
    end
    space.best_dual_bound = is_minimization(space.ctx) ?
        minimum(values(bounds)) : maximum(values(bounds))
    return
end

function TreeSearch.branch!(space::BPSpace, node)
    primal_values = get_primal_solution(space.backend)

    result = select_branching_variable(
        space.branching_strategy, space, node, primal_values
    )
    isnothing(result) && return typeof(node)[]
    if result === :node_infeasible
        @debug "branch!: node infeasible (detected by strategy)"
        return typeof(node)[]
    end
    orig_var, x_val = result

    cg_output = node.user_data.cg_output
    db = if isnothing(cg_output) ||
            isnothing(cg_output.incumbent_dual_bound)
        node.dual_bound
    else
        cg_output.incumbent_dual_bound
    end

    children, cut_info = create_branching_children(
        space.id_counter, node, orig_var, x_val,
        space.ctx, db, space.cut_tracker
    )
    for (cut_id, ov) in cut_info
        space.branching_cut_info[cut_id] = ov
    end
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
    return is_minimization(s.ctx)
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
    return length(bp_pool(s.ctx).by_column_var)
end

function TreeSearch.ts_active_columns(s::BPSpace)
    pool = bp_pool(s.ctx)
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

# ── Entry point ──────────────────────────────────────────────────────────

"""
    run_branch_and_price(ctx; strategy, node_limit, tol,
                         rmp_time_limit, rmp_heuristic,
                         separator, max_cut_rounds,
                         log_level, dot_file) -> BPOutput

Run the branch-and-price algorithm using column generation at each
node and most-fractional branching on original variables.

# Arguments
- `ctx`: Column generation context (`ColGenContext` or `ColGenLoggerContext`).
- `strategy`: Tree search strategy (default: `DepthFirstStrategy()`).
- `node_limit::Int`: Maximum nodes to explore (default: 10000).
- `tol::Float64`: Numerical tolerance (default: 1e-6).
- `rmp_time_limit::Float64`: Time limit in seconds for restricted master
  IP heuristic at each node (default: 60.0).
- `rmp_heuristic::Bool`: Run the restricted master IP heuristic at each
  node to find feasible solutions (default: true).
- `separator`: Robust cut separator (default: `nothing`).
- `max_cut_rounds::Int`: Maximum cut separation rounds per node
  (default: 0).
- `min_gap_improvement::Float64`: Minimum relative gap improvement
  to continue cut rounds (default: 0.01).
- `log_level::Int`: Logging verbosity (0 = off, 1 = table,
  2 = BaPCod-style verbose). Default: 0.
- `dot_file::Union{Nothing,String}`: Path for Graphviz `.dot` tree output
  (default: `nothing` — no dot file written).
"""
function run_branch_and_price(
    ctx::Union{ColGen.ColGenContext,ColGen.ColGenLoggerContext};
    strategy = TreeSearch.DepthFirstStrategy(),
    node_limit::Int = 10_000,
    tol::Float64 = 1e-6,
    rmp_time_limit::Float64 = 60.0,
    rmp_heuristic::Bool = true,
    separator::Union{Nothing,AbstractCutSeparator} = nothing,
    max_cut_rounds::Int = 0,
    min_gap_improvement::Float64 = 0.01,
    branching_strategy::AbstractBranchingStrategy = MostFractionalBranching(),
    log_level::Int = 0,
    dot_file::Union{Nothing,String} = nothing
)
    space = BPSpace(
        ctx;
        node_limit = node_limit,
        tol = tol,
        rmp_time_limit = rmp_time_limit,
        rmp_heuristic = rmp_heuristic,
        separator = separator,
        max_cut_rounds = max_cut_rounds,
        min_gap_improvement = min_gap_improvement,
        branching_strategy = branching_strategy
    )
    evaluator = BPEvaluator()
    if !isnothing(dot_file)
        dot_ctx = BPDotLoggerContext(space, evaluator, dot_file)
        return TreeSearch.search(strategy, dot_ctx)
    end
    if log_level > 0
        ts_ctx = TreeSearch.TreeSearchLoggerContext(
            space, evaluator, log_level
        )
        return TreeSearch.search(strategy, ts_ctx)
    end
    return TreeSearch.search(strategy, space, evaluator)
end
