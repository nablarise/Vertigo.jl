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
backend, and domain change tracking.
"""
mutable struct BPSpace <: TreeSearch.AbstractSearchSpace
    ctx::Any
    backend::Any
    domain_helper::MathOptState.DomainChangeTrackerHelper
    id_counter::TreeSearch.NodeIdCounter
    incumbent::Union{Nothing,ColGen.ProjectedIpPrimalSol}
    last_ip_incumbent::Union{Nothing,ColGen.ProjectedIpPrimalSol}
    best_dual_bound::Float64
    open_node_bounds::Dict{Int,Float64}
    nodes_explored::Int
    node_limit::Int
    tol::Float64
end

"""
    BPSpace(ctx; node_limit=10_000, tol=1e-6)

Create a branch-and-price search space from a column generation
context. Registers existing variable bound constraints for tracking.
"""
function BPSpace(
    ctx::Union{ColGen.ColGenContext,ColGen.ColGenLoggerContext};
    node_limit::Int = 10_000,
    tol::Float64 = 1e-6
)
    master = bp_master_model(ctx)
    tracker = MathOptState.DomainChangeTracker()
    domain_helper = MathOptState.transform_model!(
        tracker, master
    )
    return BPSpace(
        ctx, master, domain_helper,
        TreeSearch.NodeIdCounter(),
        nothing, nothing,
        ColGen.is_minimization(ctx) ? -Inf : Inf,
        Dict{Int,Float64}(),
        0, node_limit, tol
    )
end

# ── TreeSearch interface ─────────────────────────────────────────────────

function TreeSearch.new_root(space::BPSpace)
    node = TreeSearch.root_node(
        space.id_counter,
        MathOptState.DomainChangeDiff(),
        MathOptState.DomainChangeDiff(),
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
        space.backend, space.domain_helper
    )
    TreeSearch.transition_to!(current, next, fwd!, bwd!)
    return
end

function _recompute_global_dual_bound!(space::BPSpace)
    bounds = space.open_node_bounds
    if isempty(bounds)
        space.best_dual_bound = ColGen.is_minimization(space.ctx) ?
            -Inf : Inf
        return
    end
    space.best_dual_bound = ColGen.is_minimization(space.ctx) ?
        minimum(values(bounds)) : maximum(values(bounds))
    return
end

function TreeSearch.branch!(space::BPSpace, node)
    primal_values = Dict{MOI.VariableIndex,Float64}()
    for v in MOI.get(space.backend, MOI.ListOfVariableIndices())
        val = try
            MOI.get(space.backend, MOI.VariablePrimal(), v)
        catch
            0.0
        end
        primal_values[v] = val
    end

    branch_var, sp_id = most_fractional_column(
        space.ctx, primal_values; tol = space.tol
    )
    isnothing(branch_var) && return typeof(node)[]

    branch_val = primal_values[branch_var]
    cg_output = node.user_data.cg_output
    db = if isnothing(cg_output) ||
            isnothing(cg_output.incumbent_dual_bound)
        node.dual_bound
    else
        cg_output.incumbent_dual_bound
    end

    children = create_branching_children(
        space.id_counter, node, branch_var, branch_val,
        sp_id, space.ctx, db
    )
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
    return ColGen.is_minimization(s.ctx)
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

# ── Entry point ──────────────────────────────────────────────────────────

"""
    run_branch_and_price(ctx; strategy, node_limit, tol, log) -> BPOutput

Run the branch-and-price algorithm using column generation at each
node and most-fractional branching on column variables.

# Arguments
- `ctx`: Column generation context (`ColGenContext` or `ColGenLoggerContext`).
- `strategy`: Tree search strategy (default: `DepthFirstStrategy()`).
- `node_limit::Int`: Maximum nodes to explore (default: 10000).
- `tol::Float64`: Numerical tolerance (default: 1e-6).
- `log::Bool`: Enable VERTIGO-styled per-node logging (default: false).
"""
function run_branch_and_price(
    ctx::Union{ColGen.ColGenContext,ColGen.ColGenLoggerContext};
    strategy = TreeSearch.DepthFirstStrategy(),
    node_limit::Int = 10_000,
    tol::Float64 = 1e-6,
    log::Bool = false
)
    space = BPSpace(ctx; node_limit = node_limit, tol = tol)
    evaluator = BPEvaluator()
    if log
        ts_ctx = TreeSearch.TreeSearchLoggerContext(space, evaluator)
        return TreeSearch.search(strategy, ts_ctx)
    end
    return TreeSearch.search(strategy, space, evaluator)
end
