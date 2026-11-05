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

# ── Node view accessors ──────────────────────────────────────────────────
# Stable, read-only surface for user code (e.g. a priority function) over the
# BCP data of a node. `NodeView` itself comes from `TreeSearch`.

"""
    parent_lp_obj(node::NodeView) -> Union{Nothing,Float64}

Master LP value of the parent node, or `nothing` for the root.
"""
parent_lp_obj(node::TreeSearch.NodeView) = node.user_data.parent_lp_obj

"""
    branching_var(node::NodeView)

Original variable branched on to create `node`, or `nothing` for the root.
"""
branching_var(node::TreeSearch.NodeView) = node.user_data.branching_var

"""
    branching_direction(node::NodeView) -> Union{Nothing,BranchingDirection}

`branch_down` or `branch_up`, or `nothing` for the root.
"""
branching_direction(node::TreeSearch.NodeView) = node.user_data.branching_direction

"""
    branching_frac(node::NodeView) -> Union{Nothing,Float64}

Fractional part of the branching variable value at the parent, or `nothing`
for the root.
"""
branching_frac(node::TreeSearch.NodeView) = node.user_data.branching_frac

# ── Branch-cut-price workspace ───────────────────────────────────────────

"""
    BranchCutPriceWorkspace <: TreeSearch.AbstractSearchSpace

Branch-and-price runtime state. Wraps the column generation workspace,
the MOI master backend, domain/cut tracking, branching metadata, and
the BCP knobs flattened from the originating
[`BranchCutPriceConfig`](@ref). It is the search space passed to
`TreeSearch.search`.

Construct it from a decomposition and a config — symmetric to
[`ColGen.ColGenWorkspace`](@ref):

```julia
ws = BranchCutPriceWorkspace(decomp, BranchCutPriceConfig(...))
output = run_branch_and_price(ws)
```
"""
mutable struct BranchCutPriceWorkspace{Ws,B} <: TreeSearch.AbstractSearchSpace
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
    root_lp_value::Union{Float64,Nothing}
    node_limit::Int
    tolerance::Float64
    rmp_heuristic_time_limit::Float64
    rmp_use_heuristic::Bool
    time_limit::Float64
    deadline::Float64      # absolute `time()` of the end of the search
    cut_config::CutSeparationConfig
    total_cuts_separated::Int
    branching_strategy::AbstractBranchingStrategy
    log_level::Int
    node_select_strategy::TreeSearch.AbstractSearchStrategy
    dot_file::Union{Nothing,String}
end

"""
    BranchCutPriceWorkspace(decomp, config::BranchCutPriceConfig)

Build a `BranchCutPriceWorkspace` from a Dantzig–Wolfe decomposition
and a [`BranchCutPriceConfig`](@ref). Internally constructs a
`ColGenWorkspace` from `config.cg_config` (optionally wrapped in a
`ColGenLoggerWorkspace` when `config.cg_config.log_level > 0`), wires the
branching logger context based on `config.log_level`, and registers
domain/cut trackers on the master.
"""
function BranchCutPriceWorkspace(decomp, config::BranchCutPriceConfig)
    inner = ColGen.ColGenWorkspace(decomp, config.cg_config)
    cg_ws = if config.cg_config.log_level > 0
        ColGen.ColGenLoggerWorkspace(inner; log_level=config.cg_config.log_level)
    else
        inner
    end

    branching_ctx = if config.log_level > 0
        BranchingLoggerContext(; log_level=config.log_level)
    else
        DefaultBranchingContext()
    end
    effective_strategy = _branching_strategy_with_ctx(
        config.branching_strategy, branching_ctx
    )

    master = bp_master_model(cg_ws)
    tracker = MathOptState.DomainChangeTracker()
    domain_helper = MathOptState.transform_model!(tracker, master)
    cut_tracker = MathOptState.LocalCutTracker()
    cut_helper = MathOptState.transform_model!(cut_tracker, master)

    return BranchCutPriceWorkspace(
        cg_ws,
        master,
        domain_helper,
        cut_tracker,
        cut_helper,
        Dict{Int,Any}(),
        TreeSearch.NodeIdCounter(),
        nothing,
        nothing,
        is_minimization(cg_ws) ? -Inf : Inf,
        Dict{Int,Float64}(),
        0,
        nothing,
        config.node_limit,
        config.tolerance,
        config.rmp_heuristic_time_limit,
        config.rmp_use_heuristic,
        config.time_limit,
        Inf,
        config.cut_config,
        0,
        effective_strategy,
        config.log_level,
        config.node_select_strategy,
        config.dot_file,
    )
end

# Default: leave the strategy as-is. Specialized below for strategies
# that hold a `BranchingContext`.
_branching_strategy_with_ctx(s::AbstractBranchingStrategy, _) = s

function _branching_strategy_with_ctx(s::MultiPhaseStrongBranching, ctx::BranchingContext)
    return with_branching_ctx(s, ctx)
end

# ── TreeSearch interface ─────────────────────────────────────────────────

function TreeSearch.new_root(space::BranchCutPriceWorkspace)
    empty_fwd = (MathOptState.DomainChangeDiff(), MathOptState.LocalCutChangeDiff())
    empty_bwd = (MathOptState.DomainChangeDiff(), MathOptState.LocalCutChangeDiff())
    node = TreeSearch.root_node(space.id_counter, empty_fwd, empty_bwd, BPNodeData())
    space.open_node_bounds[node.id] = node.dual_bound
    return node
end

_time_is_up(space::BranchCutPriceWorkspace) = time() >= space.deadline

function TreeSearch.stop(space::BranchCutPriceWorkspace, _)
    return space.nodes_explored >= space.node_limit || _time_is_up(space)
end

# Why a search that is not proven optimal has stopped.
function _stop_status(space::BranchCutPriceWorkspace)
    return _time_is_up(space) ? :time_limit : :node_limit
end

function TreeSearch.output(space::BranchCutPriceWorkspace)
    status = if !isnothing(space.incumbent)
        db = space.best_dual_bound
        ub = space.incumbent.obj_value
        if abs(ub - db) < space.tolerance
            :optimal
        else
            _stop_status(space)
        end
    elseif space.nodes_explored >= space.node_limit || _time_is_up(space)
        _stop_status(space)
    else
        :infeasible
    end
    return BPOutput(
        status,
        space.incumbent,
        space.best_dual_bound,
        space.nodes_explored,
        space.root_lp_value,
    )
end

function TreeSearch.transition!(space::BranchCutPriceWorkspace, current, next)
    fwd!, bwd! = MathOptState.make_transition_callbacks(
        space.backend, (space.domain_helper, space.cut_helper)
    )
    TreeSearch.transition_to!(current, next, fwd!, bwd!)
    return nothing
end

function _recompute_global_dual_bound!(space::BranchCutPriceWorkspace)
    bounds = space.open_node_bounds
    if isempty(bounds)
        # No open node left: the incumbent is proved optimal.
        space.best_dual_bound = if !isnothing(space.incumbent)
            space.incumbent.obj_value
        else
            is_minimization(space.ws) ? -Inf : Inf
        end
        return nothing
    end
    space.best_dual_bound =
        is_minimization(space.ws) ? minimum(values(bounds)) : maximum(values(bounds))
    return nothing
end

function TreeSearch.branch!(space::BranchCutPriceWorkspace, node)
    primal_values = get_primal_solution(space.backend)

    # The node leaves the open set: its children (if any) replace it.
    delete!(space.open_node_bounds, node.id)

    result = select_branching_variable(space.branching_strategy, space, node, primal_values)
    if result.status == all_integral || result.status == node_infeasible
        _recompute_global_dual_bound!(space)
        return typeof(node)[]
    end
    orig_var = result.orig_var
    x_val = result.value

    cg_output = node.user_data.cg_output
    db = if isnothing(cg_output) || isnothing(cg_output.incumbent_dual_bound)
        node.dual_bound
    else
        cg_output.incumbent_dual_bound
    end

    parent_lp = if !isnothing(node.user_data) && !isnothing(node.user_data.cg_output)
        node.user_data.cg_output.master_lp_obj
    else
        nothing
    end
    branching_frac = x_val - floor(x_val)

    children, cut_info = create_branching_children(
        space.id_counter, node, orig_var, x_val, space.ws, db, space.cut_tracker
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
    ::BranchCutPriceWorkspace, ::TreeSearch.SearchNode
)
    # Incumbent already updated in evaluate!
    return nothing
end

# ── TreeSearch logger protocol ───────────────────────────────────────────

function TreeSearch.ts_incumbent_value(s::BranchCutPriceWorkspace)
    return isnothing(s.incumbent) ? nothing : s.incumbent.obj_value
end

function TreeSearch.ts_best_dual_bound(s::BranchCutPriceWorkspace)
    return s.best_dual_bound
end

function TreeSearch.ts_is_minimization(s::BranchCutPriceWorkspace)
    return is_minimization(s.ws)
end

function TreeSearch.ts_nodes_explored(s::BranchCutPriceWorkspace)
    return s.nodes_explored
end

function TreeSearch.ts_search_status_message(s::BranchCutPriceWorkspace)
    inc = s.incumbent
    if !isnothing(inc)
        db = s.best_dual_bound
        if abs(inc.obj_value - db) < s.tolerance
            return "Optimal solution found."
        end
    end
    if s.nodes_explored >= s.node_limit
        return "Node limit reached."
    end
    return isnothing(inc) ? "No feasible solution found." : "Search complete."
end

function TreeSearch.ts_open_node_count(s::BranchCutPriceWorkspace)
    return length(s.open_node_bounds)
end

function TreeSearch.ts_total_columns(s::BranchCutPriceWorkspace)
    return length(bp_pool(s.ws).by_column_var)
end

function TreeSearch.ts_active_columns(s::BranchCutPriceWorkspace)
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

function TreeSearch.ts_total_cuts(s::BranchCutPriceWorkspace)
    return s.total_cuts_separated
end

function TreeSearch.ts_branching_description(
    s::BranchCutPriceWorkspace, node::TreeSearch.SearchNode
)
    if TreeSearch.is_root(node)
        return nothing
    end
    _, cut_diff = node.local_forward_diff
    if isempty(cut_diff.add_cuts)
        return nothing
    end
    cut = cut_diff.add_cuts[1].cut
    orig_var = get(s.branching_cut_info, cut.id, nothing)
    if isnothing(orig_var)
        return nothing
    end
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
    run_branch_and_price(ws::BranchCutPriceWorkspace) -> BPOutput

Run the branch-and-price algorithm described by `ws`.
"""
function run_branch_and_price(ws::BranchCutPriceWorkspace)
    ws.deadline = time() + ws.time_limit
    ColGen.set_deadline!(ws.ws, ws.deadline)
    evaluator = BPEvaluator()
    if !isnothing(ws.dot_file)
        dot_ctx = BPDotLoggerContext(ws, evaluator, ws.dot_file)
        return TreeSearch.search(ws.node_select_strategy, dot_ctx)
    end
    if ws.log_level > 0
        ts_ctx = TreeSearch.TreeSearchLoggerContext(ws, evaluator, ws.log_level)
        return TreeSearch.search(ws.node_select_strategy, ts_ctx)
    end
    return TreeSearch.search(ws.node_select_strategy, ws, evaluator)
end
