# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

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

"""
    ts_open_node_count(space::AbstractSearchSpace)

Return the number of open (unexplored) nodes, or `0` if unknown.
"""
ts_open_node_count(::AbstractSearchSpace) = 0

"""
    ts_total_columns(space::AbstractSearchSpace)

Return the total number of columns in the pool, or `0` if unknown.
"""
ts_total_columns(::AbstractSearchSpace) = 0

"""
    ts_active_columns(space::AbstractSearchSpace)

Return the number of active columns in the master, or `0` if unknown.
"""
ts_active_columns(::AbstractSearchSpace) = 0

"""
    ts_total_cuts(space::AbstractSearchSpace)

Return the total number of separated cuts added so far, or `0`.
"""
ts_total_cuts(::AbstractSearchSpace) = 0

"""
    ts_branching_description(space::AbstractSearchSpace, node::SearchNode)

Return a `String` describing the branching constraint that created
`node`, or `nothing` for the root node.
"""
ts_branching_description(::AbstractSearchSpace, ::SearchNode) = nothing

# ────────────────────────────────────────────────────────────────────────────────
# Part B: _NodeLoggerEvaluator — wraps evaluator to log after evaluate!
# ────────────────────────────────────────────────────────────────────────────────

struct _NodeLoggerEvaluator{E<:AbstractNodeEvaluator} <: AbstractNodeEvaluator
    inner::E
    space::AbstractSearchSpace
    start_time::Float64
    log_level::Int
end

function evaluate!(
    wrapper::_NodeLoggerEvaluator, space::AbstractSearchSpace,
    node::SearchNode
)
    if wrapper.log_level >= 2
        _print_verbose_banner(
            wrapper.space, node, wrapper.start_time
        )
    end
    status = evaluate!(wrapper.inner, space, node)
    if wrapper.log_level >= 2
        _print_verbose_node_footer(
            wrapper.space, node, status, wrapper.start_time
        )
    else
        _print_ts_row(
            wrapper.space, node, status, wrapper.start_time
        )
    end
    return status
end

# ────────────────────────────────────────────────────────────────────────────────
# Part C1: Level 2 (BaPCod-style) verbose formatting
# ────────────────────────────────────────────────────────────────────────────────

const _VERBOSE_SEP = "*" ^ 80

function _fmt_elapsed(start_time)
    secs = time() - start_time
    h = floor(Int, secs / 3600)
    secs -= h * 3600
    m = floor(Int, secs / 60)
    secs -= m * 60
    s = floor(Int, secs)
    t = floor(Int, (secs - s) * 100)
    return "$(h)h$(m)m$(s)s$(t)t"
end

function _fmt_bound_verbose(x)
    isnothing(x) && return "N/A"
    isinf(x) && return x < 0 ? "-1e+12" : "1e+06"
    return string(round(x; digits=6))
end

function _print_verbose_banner(space, node, start_time)
    println(_VERBOSE_SEP)
    if is_root(node)
        println("**** BaB tree root node")
    else
        @printf("**** BaB tree node %d (depth %d)\n",
            node.id, node.depth)
        desc = ts_branching_description(space, node)
        if !isnothing(desc)
            println("**** Branching constraint: ", desc)
        end
    end
    db = ts_best_dual_bound(space)
    inc = ts_incumbent_value(space)
    elapsed = _fmt_elapsed(start_time)
    @printf("**** Local DB = %s, global bounds : [ %s , %s ], TIME = %s\n",
        _fmt_bound_verbose(db),
        _fmt_bound_verbose(db),
        _fmt_bound_verbose(inc),
        elapsed)
    open_n = ts_open_node_count(space)
    total_c = ts_total_columns(space)
    active_c = ts_active_columns(space)
    cuts = ts_total_cuts(space)
    @printf("**** %d open nodes, %d columns (%d active), %d cuts\n",
        open_n, total_c, active_c, cuts)
    println(_VERBOSE_SEP)
end

function _print_verbose_node_footer(space, node, status, start_time)
    db = ts_best_dual_bound(space)
    inc = ts_incumbent_value(space)
    minimize = ts_is_minimization(space)
    elapsed = _fmt_elapsed(start_time)
    gap_str = _fmt_gap(db, inc, minimize)
    @printf("**** Node %d done (%s) -- DB = %s, Inc = %s, Gap = %s, TIME = %s\n",
        node.id, _fmt_node_status(status),
        _fmt_bound_verbose(db), _fmt_bound_verbose(inc),
        gap_str, elapsed)
    println()
end

# ────────────────────────────────────────────────────────────────────────────────
# Part C2: Table formatting (Level 1)
# ────────────────────────────────────────────────────────────────────────────────

const _TS_HDR = "    NODE   DEPTH   STATUS       BEST DB      INCUMBENT    GAP (%)   CUTS   TIME (s)"
const _TS_SEP = "  ------   -----   -------   -----------   -----------   --------   ----   --------"

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
    cuts = ts_total_cuts(space)
    t = time() - start_time
    @printf("  %6d   %5d   %-6s   %s   %s   %s   %4d   %8.2f\n",
        node.id, node.depth, _fmt_node_status(status),
        _fmt_val(db), _fmt_val(inc),
        _fmt_gap(db, inc, minimize), cuts, t)
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
    log_level::Int
end

function TreeSearchLoggerContext(space::S, evaluator::E) where {
    S<:AbstractSearchSpace,E<:AbstractNodeEvaluator
}
    return TreeSearchLoggerContext{S,E}(space, evaluator, 1)
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
    if ctx.log_level <= 1
        println(_TS_HDR)
        println(_TS_SEP)
    end
    wrapper = _NodeLoggerEvaluator(
        ctx.evaluator, ctx.space, start_time, ctx.log_level
    )
    result = search(strategy, ctx.space, wrapper)
    _print_ts_footer(ctx.space, start_time, ctx.log_level)
    return result
end

# ────────────────────────────────────────────────────────────────────────────────
# Part F: Footer
# ────────────────────────────────────────────────────────────────────────────────

function _print_ts_footer(space, start_time, log_level::Int=1)
    t = time() - start_time
    msg = ts_search_status_message(space)
    db  = ts_best_dual_bound(space)
    inc = ts_incumbent_value(space)
    minimize = ts_is_minimization(space)
    nodes = ts_nodes_explored(space)
    cuts = ts_total_cuts(space)
    if log_level >= 2
        elapsed = _fmt_elapsed(start_time)
        println(_VERBOSE_SEP)
        @printf("**** %s\n", msg)
        @printf("**** DB = %s, Inc = %s, Gap = %s, Nodes = %d, Cuts = %d, TIME = %s\n",
            _fmt_bound_verbose(db),
            _fmt_bound_verbose(inc),
            _fmt_gap(db, inc, minimize),
            nodes, cuts, elapsed)
        println(_VERBOSE_SEP)
    else
        inc_str = isnothing(inc) ? "N/A" : @sprintf("%.3e", inc)
        db_str  = isnothing(db)  ? "N/A" : @sprintf("%.3e", db)
        gap_str = _fmt_gap(db, inc, minimize)
        println()
        @printf("[STATUS] :: %s\n", msg)
        @printf("[RESULT] :: Incumbent: %s | Dual: %s | Gap: %s | Nodes: %d | Cuts: %d | Time: %.1fs\n",
            inc_str, db_str, gap_str, nodes, cuts, t)
    end
end
