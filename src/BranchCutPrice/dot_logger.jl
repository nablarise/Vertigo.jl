# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    DotNodeStatus

Visual status of a search node in the Graphviz dot output.

- `DotBranched`: Node was branched into children.
- `DotPrunedInfeasible`: Node pruned due to infeasibility.
- `DotPrunedBound`: Node pruned by bound.
- `DotFeasible`: IP-feasible solution found at this node.
"""
@enum DotNodeStatus begin
    DotBranched
    DotPrunedInfeasible
    DotPrunedBound
    DotFeasible
end

"""
    BPDotLoggerContext

Wraps a branch-and-price search space and evaluator to produce an
incremental Graphviz `.dot` file of the search tree.

# Fields
- `space::BPSpace`: The B&P search space.
- `evaluator::BPEvaluator`: The node evaluator.
- `filepath::String`: Output `.dot` file path.
- `branching_labels::Dict{Int,String}`: Maps child node ID to edge label.
"""
mutable struct BPDotLoggerContext
    space::BPSpace
    evaluator::BPEvaluator
    filepath::String
    branching_labels::Dict{Int,String}
end

function BPDotLoggerContext(
    space::BPSpace,
    evaluator::BPEvaluator,
    filepath::String
)
    return BPDotLoggerContext(
        space, evaluator, filepath, Dict{Int,String}()
    )
end

# ── Status derivation ────────────────────────────────────────────────────

function _derive_dot_status(
    ts_status::TreeSearch.NodeStatus,
    cg_output::Union{Nothing,ColGen.ColGenOutput}
)
    if ts_status == TreeSearch.FEASIBLE
        return DotFeasible
    elseif ts_status == TreeSearch.CUTOFF
        if !isnothing(cg_output) && (
            cg_output.status == ColGen.master_infeasible ||
            cg_output.status == ColGen.subproblem_infeasible ||
            cg_output.status == ColGen.ip_pruned
        )
            return DotPrunedInfeasible
        end
        return DotPrunedBound
    end
    return DotBranched
end

# ── Dot formatting ───────────────────────────────────────────────────────

function _dot_shape_and_color(s::DotNodeStatus)
    s == DotBranched && return ("ellipse", "lightblue")
    s == DotPrunedInfeasible && return ("box", "gray")
    s == DotPrunedBound && return ("box", "lightcoral")
    return ("doubleoctagon", "gold")
end

function _dot_status_text(s::DotNodeStatus)
    s == DotBranched && return "branched"
    s == DotPrunedInfeasible && return "infeasible"
    s == DotPrunedBound && return "pruned (bound)"
    return "IP feasible"
end

function _write_dot_node(io::IO, node, dot_status::DotNodeStatus,
                         dual_bound, incumbent_value)
    shape, color = _dot_shape_and_color(dot_status)
    db_str = isnothing(dual_bound) ? "N/A" :
        @sprintf("%.2f", dual_bound)
    ip_str = isnothing(incumbent_value) ? "\u2014" :
        @sprintf("%.2f", incumbent_value)
    status_str = _dot_status_text(dot_status)
    label = "Node $(node.id)\\nDB = $(db_str)" *
        "\\nIP* = $(ip_str)\\n$(status_str)"
    println(
        io,
        "  n$(node.id) [label=\"$(label)\", " *
        "shape=$(shape), fillcolor=$(color), style=filled];"
    )
end

function _write_dot_edge(io::IO, parent_id::Int, child_id::Int,
                         label::String)
    escaped = replace(label, "\"" => "\\\"")
    println(
        io,
        "  n$(parent_id) -> n$(child_id) " *
        "[label=\"$(escaped)\"];"
    )
end

# ── Custom search loop with dot logging ──────────────────────────────────

"""
    TreeSearch.search(strategy, ctx::BPDotLoggerContext)

Run tree search and write an incremental Graphviz `.dot` file with
color-coded nodes and branching-decision edge labels.
"""
function TreeSearch.search(
    strategy::TreeSearch.AbstractSearchStrategy,
    ctx::BPDotLoggerContext
)
    space = ctx.space
    evaluator = ctx.evaluator

    open(ctx.filepath, "w") do io
        println(io, "digraph BCP_Tree {")
        println(io, "  rankdir=TB;")
        println(io, "  node [fontsize=10];")
        println(io, "  edge [fontsize=9];")

        root = TreeSearch.new_root(space)
        container = TreeSearch.init_container(strategy, root)
        current_node = root
        is_first = true

        while !TreeSearch.stop(space, container) &&
              !TreeSearch.is_empty_container(container)
            next_node = TreeSearch.select_node!(strategy, container)

            if is_first
                is_first = false
            else
                TreeSearch.transition!(space, current_node, next_node)
            end

            status = TreeSearch.evaluate!(evaluator, space, next_node)

            # Derive dot status and dual bound
            cg_out = next_node.user_data.cg_output
            dot_status = _derive_dot_status(status, cg_out)
            dual_bound = isnothing(cg_out) ? nothing :
                cg_out.incumbent_dual_bound

            inc_val = isnothing(space.incumbent) ? nothing :
                space.incumbent.obj_value
            _write_dot_node(
                io, next_node, dot_status, dual_bound, inc_val
            )

            # Edge from parent
            if !isnothing(next_node.parent)
                edge_label = get(
                    ctx.branching_labels, next_node.id, ""
                )
                _write_dot_edge(
                    io, next_node.parent.id,
                    next_node.id, edge_label
                )
            end

            if status == TreeSearch.BRANCH
                children = TreeSearch.branch!(space, next_node)
                _record_branching_labels!(ctx, children)
                TreeSearch.insert_children!(
                    strategy, container, children
                )
            elseif status == TreeSearch.FEASIBLE
                TreeSearch.on_feasible_solution!(space, next_node)
                TreeSearch.prune!(space, container)
            end

            current_node = next_node
        end

        println(io, "}")
    end

    return TreeSearch.output(space)
end

# ── Branching label extraction ───────────────────────────────────────────

function _record_branching_labels!(ctx::BPDotLoggerContext, children)
    for child in children
        _, cut_diff = child.local_forward_diff
        isempty(cut_diff.add_cuts) && continue
        cut = cut_diff.add_cuts[1].cut
        orig_var = get(
            ctx.space.branching_cut_info, cut.id, nothing
        )
        isnothing(orig_var) && continue
        if cut.set isa MOI.LessThan
            rhs = cut.set.upper
            ctx.branching_labels[child.id] =
                "$(orig_var) <= $(Int(rhs))"
        elseif cut.set isa MOI.GreaterThan
            rhs = cut.set.lower
            ctx.branching_labels[child.id] =
                "$(orig_var) >= $(Int(rhs))"
        end
    end
end
