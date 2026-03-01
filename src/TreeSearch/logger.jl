# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────
# Part A: Optional protocol methods for richer logging
# ────────────────────────────────────────────────────────────────────────────────

"""
    ts_incumbent_value(space::AbstractSearchSpace)

Return the current incumbent objective value, or `nothing` if none.
"""
ts_incumbent_value(::AbstractSearchSpace) = nothing

"""
    ts_best_dual_bound(space::AbstractSearchSpace)

Return the current best dual bound, or `nothing` if none.
"""
ts_best_dual_bound(::AbstractSearchSpace) = nothing

"""
    ts_is_minimization(space::AbstractSearchSpace)

Return `true` if the search minimizes (default).
"""
ts_is_minimization(::AbstractSearchSpace) = true

"""
    ts_nodes_explored(space::AbstractSearchSpace)

Return the number of nodes explored so far.
"""
ts_nodes_explored(::AbstractSearchSpace) = 0

"""
    ts_search_status_message(space::AbstractSearchSpace)

Return a human-readable status message for the footer.
"""
ts_search_status_message(::AbstractSearchSpace) = "Search complete."

# ────────────────────────────────────────────────────────────────────────────────
# Part B: _NodeLoggerEvaluator — wraps evaluator to log after evaluate!
# ────────────────────────────────────────────────────────────────────────────────

struct _NodeLoggerEvaluator{E<:AbstractNodeEvaluator} <: AbstractNodeEvaluator
    inner::E
    space::AbstractSearchSpace
    start_time::Float64
end

function evaluate!(
    wrapper::_NodeLoggerEvaluator, space::AbstractSearchSpace,
    node::SearchNode
)
    status = evaluate!(wrapper.inner, space, node)
    _print_ts_row(wrapper.space, node, status, wrapper.start_time)
    return status
end

# ────────────────────────────────────────────────────────────────────────────────
# Part C: Table formatting
# ────────────────────────────────────────────────────────────────────────────────

const _TS_HDR = "    NODE   DEPTH   STATUS       BEST DB      INCUMBENT    GAP (%)   TIME (s)"
const _TS_SEP = "  ------   -----   -------   -----------   -----------   --------   --------"

_fmt_val(::Nothing) = "        N/A"
function _fmt_val(x::Float64)
    isinf(x) && return x < 0 ? "       -inf" : "        inf"
    return @sprintf("%11.5e", x)
end

function _fmt_node_status(status::NodeStatus)
    status == BRANCH  && return "BRANCH"
    status == FEASIBLE && return "FEASBL"
    return "CUTOFF"
end

function _fmt_gap(db, inc, minimize::Bool)
    isnothing(db) && return "     N/A"
    isnothing(inc) && return "     N/A"
    isinf(db) && return "     N/A"
    isinf(inc) && return "     N/A"
    denom = abs(inc)
    denom < 1e-10 && return "     N/A"
    gap = minimize ? (inc - db) / denom : (db - inc) / denom
    gap = abs(gap) * 100.0
    return @sprintf("%7.2f%%", gap)
end

function _print_ts_row(space, node, status, start_time)
    db  = ts_best_dual_bound(space)
    inc = ts_incumbent_value(space)
    minimize = ts_is_minimization(space)
    t = time() - start_time
    @printf("  %6d   %5d   %-6s   %s   %s   %s   %8.2f\n",
        node.id, node.depth, _fmt_node_status(status),
        _fmt_val(db), _fmt_val(inc),
        _fmt_gap(db, inc, minimize), t)
end

# ────────────────────────────────────────────────────────────────────────────────
# Part D: TreeSearchLoggerContext
# ────────────────────────────────────────────────────────────────────────────────

"""
    TreeSearchLoggerContext{S,E}

Wraps a search space and evaluator to add VERTIGO-styled per-node
logging during tree search.

# Arguments
- `space::S`: The search space.
- `evaluator::E`: The node evaluator.
"""
struct TreeSearchLoggerContext{
    S<:AbstractSearchSpace,E<:AbstractNodeEvaluator
}
    space::S
    evaluator::E
end

# ────────────────────────────────────────────────────────────────────────────────
# Part E: search entry point
# ────────────────────────────────────────────────────────────────────────────────

"""
    search(strategy, ctx::TreeSearchLoggerContext)

Run tree search with VERTIGO-styled per-node logging.
Prints a header, delegates to the 3-arg `search`, then prints a footer.
"""
function search(
    strategy::AbstractSearchStrategy,
    ctx::TreeSearchLoggerContext
)
    start_time = time()
    println(_TS_HDR)
    println(_TS_SEP)
    wrapper = _NodeLoggerEvaluator(ctx.evaluator, ctx.space, start_time)
    result = search(strategy, ctx.space, wrapper)
    _print_ts_footer(ctx.space, start_time)
    return result
end

# ────────────────────────────────────────────────────────────────────────────────
# Part F: Footer
# ────────────────────────────────────────────────────────────────────────────────

function _print_ts_footer(space, start_time)
    t = time() - start_time
    msg = ts_search_status_message(space)
    println()
    @printf("[STATUS] :: %s\n", msg)
    db  = ts_best_dual_bound(space)
    inc = ts_incumbent_value(space)
    minimize = ts_is_minimization(space)
    nodes = ts_nodes_explored(space)
    inc_str = isnothing(inc) ? "N/A" : @sprintf("%.3e", inc)
    db_str  = isnothing(db)  ? "N/A" : @sprintf("%.3e", db)
    gap_str = _fmt_gap(db, inc, minimize)
    @printf("[RESULT] :: Incumbent: %s | Dual: %s | Gap: %s | Nodes: %d | Time: %.1fs\n",
        inc_str, db_str, gap_str, nodes, t)
end
