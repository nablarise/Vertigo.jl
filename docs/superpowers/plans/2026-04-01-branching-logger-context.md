<!-- Copyright (c) 2025 Nablarise. All rights reserved. -->

# Branching Logger Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract all logging from the branching kernel into a `BranchingLoggerContext` that wraps the default `BranchingContext`, following the ColGen logger pattern.

**Architecture:** The branching kernel (`kernel.jl`, `strong_branching.jl`) calls hook functions on a context argument at well-defined points. `BranchingContext` provides no-op defaults. `BranchingLoggerContext` wraps it and overrides hooks to emit formatted output. `MultiPhaseStrongBranching` holds a context; `BPSpace` constructs the right type based on `log_level`.

**Tech Stack:** Julia, MathOptInterface, Printf

---

### Task 1: Create `BranchingContext` with no-op hook stubs

**Files:**
- Create: `src/Branching/context.jl`
- Modify: `src/Branching/Branching.jl:15-25` (add include)

- [ ] **Step 1: Create `src/Branching/context.jl`**

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    BranchingContext

Default branching context. All hook functions are no-ops.
Subtype or wrap to add logging, profiling, or other
cross-cutting concerns.
"""
struct BranchingContext end

# ── Hook stubs (no-ops) ─────────────────────────────────────

function before_branching_selection(
    ::BranchingContext, candidates, phases
)
    return
end

function before_probe(
    ::BranchingContext, phase, candidate, direction::Symbol
)
    return
end

function after_probe(
    ::BranchingContext, phase, candidate,
    direction::Symbol, result
)
    return
end

function after_candidate_eval(
    ::BranchingContext, phase, idx::Int, candidate,
    score::Float64, detail
)
    return
end

function on_both_infeasible(
    ::BranchingContext, phase, idx::Int, candidate
)
    return
end

function after_phase_filter(
    ::BranchingContext, label::String,
    before::Int, after::Int
)
    return
end

function after_branching_selection(
    ::BranchingContext, candidate, score::Float64
)
    return
end
```

- [ ] **Step 2: Add include to `Branching.jl`**

Insert `include("context.jl")` after `include("pseudocosts.jl")` (line 24) and before `include("kernel.jl")` (line 25):

```julia
include("pseudocosts.jl")
include("context.jl")
include("kernel.jl")
```

- [ ] **Step 3: Add `BranchingContext` to exports**

In `Branching.jl`, add `BranchingContext` to the export list (line 28):

```julia
export AbstractBranchingPhase, LPProbePhase, CGProbePhase
export AbstractBranchingStrategy, MostFractionalBranching
export MultiPhaseStrongBranching
export AbstractBranchingRule, MostFractionalRule, LeastFractionalRule
export BranchingStatus, BranchingResult
export branching_ok, all_integral, node_infeasible
export BranchingDirection, branch_down, branch_up
export select_branching_variable, on_node_evaluated
export BranchingContext
```

- [ ] **Step 4: Verify it loads**

Run: `julia --project=. -e 'using Vertigo; println(Vertigo.Branching.BranchingContext())'`
Expected: prints `BranchingContext()` with no error.

- [ ] **Step 5: Commit**

```bash
git add src/Branching/context.jl src/Branching/Branching.jl
git commit -m "add: BranchingContext with no-op hook stubs"
```

---

### Task 2: Strip logging from `strong_branching.jl`

**Files:**
- Modify: `src/Branching/strong_branching.jl`

- [ ] **Step 1: Remove `@warn` from `_sb_delta` (line 29)**

Replace:
```julia
function _sb_delta(probe::SBProbeResult, parent_lp_obj::Float64)
    probe.is_infeasible && return Inf
    if isnothing(probe.dual_bound)
        @warn "SB probe returned no dual bound; scoring as Δ=0"
        return 0.0
    end
    return max(0.0, probe.dual_bound - parent_lp_obj)
end
```

With:
```julia
function _sb_delta(probe::SBProbeResult, parent_lp_obj::Float64)
    probe.is_infeasible && return Inf
    isnothing(probe.dual_bound) && return 0.0
    return max(0.0, probe.dual_bound - parent_lp_obj)
end
```

- [ ] **Step 2: Add context argument to `run_sb_probe` and add hook calls**

Replace:
```julia
function run_sb_probe(
    space, candidate::BranchingCandidate,
    max_cg_iterations::Int, parent_lp_obj::Float64
)
    snapshot = _capture_probe_state(space.ctx, space)
    try
        left = _run_one_direction(
            space, candidate,
            MOI.LessThan(candidate.floor_val),
            max_cg_iterations
        )
        _restore_probe_state!(space.ctx, space, snapshot)
        right = _run_one_direction(
            space, candidate,
            MOI.GreaterThan(candidate.ceil_val),
            max_cg_iterations
        )
        @debug "SB probe" candidate.orig_var left right
        return SBCandidateResult(
            candidate, parent_lp_obj, left, right
        )
    finally
        _restore_probe_state!(space.ctx, space, snapshot)
    end
end
```

With:
```julia
function run_sb_probe(
    bctx::BranchingContext, space,
    candidate::BranchingCandidate,
    max_cg_iterations::Int, parent_lp_obj::Float64
)
    snapshot = _capture_probe_state(space.ctx, space)
    try
        before_probe(bctx, nothing, candidate, :left)
        left = _run_one_direction(
            space, candidate,
            MOI.LessThan(candidate.floor_val),
            max_cg_iterations
        )
        after_probe(bctx, nothing, candidate, :left, left)
        _restore_probe_state!(space.ctx, space, snapshot)
        before_probe(bctx, nothing, candidate, :right)
        right = _run_one_direction(
            space, candidate,
            MOI.GreaterThan(candidate.ceil_val),
            max_cg_iterations
        )
        after_probe(bctx, nothing, candidate, :right, right)
        return SBCandidateResult(
            candidate, parent_lp_obj, left, right
        )
    finally
        _restore_probe_state!(space.ctx, space, snapshot)
    end
end
```

Note: The `phase` argument to `before_probe`/`after_probe` is `nothing` here because `run_sb_probe` does not know the current phase. We will pass the phase from `probe_candidate` in task 3.

- [ ] **Step 3: Delete logging helpers**

Remove these three functions entirely (lines 212-249):
- `_sb_log_header`
- `_sb_log_candidate`
- `_sb_log_selected`

Also remove `_sb_fmt_bound` (lines 217-221) — it will move to `logger.jl`.

- [ ] **Step 4: Verify it compiles**

Run: `julia --project=. -e 'using Vertigo'`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add src/Branching/strong_branching.jl
git commit -m "remove: logging from strong_branching.jl"
```

---

### Task 3: Strip logging from `kernel.jl` and thread context

**Files:**
- Modify: `src/Branching/kernel.jl`
- Modify: `src/Branching/cg_probe.jl`
- Modify: `src/Branching/lp_probe.jl`

- [ ] **Step 1: Update `run_branching_selection` signature and body**

Replace the full function (lines 111-194):

```julia
function run_branching_selection(
    bctx::BranchingContext, space, node,
    phases::Vector{<:AbstractBranchingPhase},
    pseudocosts::PseudocostTracker,
    primal_values::Dict{MOI.VariableIndex,Float64};
    max_candidates::Int = 100,
    mu::Float64 = 1.0 / 6.0,
    tol::Float64 = 1e-6
)
    ctx = space.ctx
    candidates = find_fractional_variables(
        ctx, primal_values; tol=tol
    )
    isempty(candidates) && return BranchingResult(all_integral)

    parent_lp = _get_parent_lp(node)
    if isnothing(parent_lp)
        c = first(candidates)
        return BranchingResult(c.orig_var, c.value)
    end

    current = select_initial_candidates(
        pseudocosts, candidates, max_candidates; mu=mu
    )

    before_branching_selection(bctx, current, phases)

    best_candidate = first(current)
    best_score = -Inf

    for (phase_idx, phase) in enumerate(phases)
        label = phase_label(phase)
        next_phase = phase_idx < length(phases) ?
            phases[phase_idx + 1] : nothing

        scored = Tuple{BranchingCandidate,Float64}[]
        no_improvement_count = 0

        for (idx, c) in enumerate(current)
            if stop_phase(phase, idx, best_score,
                          no_improvement_count)
                break
            end

            score = _eval_candidate(
                bctx, phase, space, pseudocosts, c,
                parent_lp, idx, mu
            )
            isnothing(score) && return BranchingResult(
                node_infeasible
            )

            push!(scored, (c, score))
            if score > best_score
                best_score = score
                best_candidate = c
                no_improvement_count = 0
            else
                no_improvement_count += 1
            end
        end

        sort!(scored; by=x -> x[2], rev=true)

        before = length(scored)
        current = filter_candidates(phase, next_phase, scored)
        if !isnothing(next_phase)
            after_phase_filter(
                bctx, label, before, length(current)
            )
        end
    end

    after_branching_selection(bctx, best_candidate, best_score)
    return BranchingResult(
        best_candidate.orig_var, best_candidate.value
    )
end
```

- [ ] **Step 2: Update `_eval_candidate`**

Replace the full function (lines 200-248):

```julia
function _eval_candidate(
    bctx::BranchingContext,
    phase::AbstractBranchingPhase, space,
    pseudocosts::PseudocostTracker,
    c::BranchingCandidate, parent_lp::Float64,
    idx::Int, mu::Float64
)
    if phase isa CGProbePhase &&
       is_reliable(pseudocosts, c)
        score = estimate_score(pseudocosts, c; mu=mu)
        after_candidate_eval(
            bctx, phase, idx, c, score, :reliable
        )
        return score
    end

    result = probe_candidate(bctx, phase, space, c, parent_lp)

    if result.left.is_infeasible &&
       result.right.is_infeasible
        on_both_infeasible(bctx, phase, idx, c)
        return nothing
    end

    score = score_candidate(phase, result; mu=mu)
    update_pseudocosts!(pseudocosts, c, result)

    after_candidate_eval(
        bctx, phase, idx, c, score, result
    )
    return score
end
```

- [ ] **Step 3: Update `select_branching_variable` for `MultiPhaseStrongBranching`**

Replace (lines 292-304):

```julia
function select_branching_variable(
    mpsb::MultiPhaseStrongBranching, space, node,
    primal_values::Dict{MOI.VariableIndex,Float64}
)
    return run_branching_selection(
        mpsb.branching_ctx, space, node,
        mpsb.phases, mpsb.pseudocosts,
        primal_values;
        max_candidates=mpsb.max_candidates,
        mu=mpsb.mu,
        tol=space.tol
    )
end
```

- [ ] **Step 4: Add `branching_ctx` field to `MultiPhaseStrongBranching`**

Replace the struct definition (lines 260-283):

```julia
struct MultiPhaseStrongBranching <: AbstractBranchingStrategy
    max_candidates::Int
    mu::Float64
    phases::Vector{AbstractBranchingPhase}
    pseudocosts::PseudocostTracker{Any}
    branching_ctx::BranchingContext

    function MultiPhaseStrongBranching(;
        max_candidates::Int = 20,
        mu::Float64 = 1.0 / 6.0,
        phases::Vector{<:AbstractBranchingPhase} = AbstractBranchingPhase[
            LPProbePhase(keep_fraction=0.25),
            CGProbePhase(max_cg_iterations=10, lookahead=8)
        ],
        reliability_threshold::Int = 8,
        branching_ctx::BranchingContext = BranchingContext()
    )
        new(
            max_candidates, mu,
            convert(Vector{AbstractBranchingPhase}, phases),
            PseudocostTracker{Any}(
                reliability_threshold=reliability_threshold
            ),
            branching_ctx
        )
    end
end
```

- [ ] **Step 5: Update `probe_candidate` in `cg_probe.jl` to thread context**

Replace `src/Branching/cg_probe.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    probe_candidate(bctx, ::CGProbePhase, space, candidate, parent_lp)

CG probe: run column generation with limited iterations in both
directions. Delegates to `run_sb_probe`.
"""
function probe_candidate(
    bctx::BranchingContext, phase::CGProbePhase, space,
    candidate::BranchingCandidate, parent_lp::Float64
)
    return run_sb_probe(
        bctx, space, candidate,
        phase.max_cg_iterations, parent_lp
    )
end
```

- [ ] **Step 6: Update `probe_candidate` in `lp_probe.jl` to thread context**

Replace the `probe_candidate` function in `src/Branching/lp_probe.jl` (lines 60-81):

```julia
"""
    probe_candidate(bctx, ::LPProbePhase, space, candidate, parent_lp)

LP probe: solve master LP only (no CG) in both directions.
Uses same state capture/restore as CG probes.
"""
function probe_candidate(
    bctx::BranchingContext, ::LPProbePhase, space,
    candidate::BranchingCandidate, parent_lp::Float64
)
    snapshot = _capture_probe_state(space.ctx, space)
    try
        before_probe(bctx, nothing, candidate, :left)
        left = _run_one_lp_direction(
            space, candidate,
            MOI.LessThan(candidate.floor_val)
        )
        after_probe(bctx, nothing, candidate, :left, left)
        _restore_probe_state!(space.ctx, space, snapshot)
        before_probe(bctx, nothing, candidate, :right)
        right = _run_one_lp_direction(
            space, candidate,
            MOI.GreaterThan(candidate.ceil_val)
        )
        after_probe(bctx, nothing, candidate, :right, right)
        return SBCandidateResult(
            candidate, parent_lp, left, right
        )
    finally
        _restore_probe_state!(space.ctx, space, snapshot)
    end
end
```

- [ ] **Step 7: Remove `@warn` from `run_branching_selection`**

The `@warn "run_branching_selection: no parent LP obj..."` (line 129-130) was removed in Step 1 — the fallback now just returns the first candidate silently.

- [ ] **Step 8: Verify it compiles**

Run: `julia --project=. -e 'using Vertigo'`
Expected: no error.

- [ ] **Step 9: Commit**

```bash
git add src/Branching/kernel.jl src/Branching/cg_probe.jl src/Branching/lp_probe.jl
git commit -m "remove: logging from branching kernel, thread BranchingContext"
```

---

### Task 4: Remove `@debug` from `space.jl`

**Files:**
- Modify: `src/BranchCutPrice/space.jl:171`

- [ ] **Step 1: Remove `@debug` from `branch!`**

Replace (line 171):
```julia
        @debug "branch!: node infeasible (detected by strategy)"
```

With nothing — just delete the line. The `return typeof(node)[]` on the next line is sufficient.

- [ ] **Step 2: Verify it compiles**

Run: `julia --project=. -e 'using Vertigo'`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add src/BranchCutPrice/space.jl
git commit -m "remove: @debug from branch! in space.jl"
```

---

### Task 5: Create `BranchingLoggerContext`

**Files:**
- Create: `src/Branching/logger.jl`
- Modify: `src/Branching/Branching.jl` (add include + export)

- [ ] **Step 1: Create `src/Branching/logger.jl`**

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ──────────────────────────────────────────────────────────────
# BRANCHING LOGGER CONTEXT
# A thin wrapper around BranchingContext that overrides hook
# functions to emit formatted terminal output.
# ──────────────────────────────────────────────────────────────

mutable struct BranchingLoggerContext <: BranchingContext
    io::IO
    log_level::Int
    t0::Float64

    function BranchingLoggerContext(;
        io::IO = stdout,
        log_level::Int = 1
    )
        return new(io, log_level, 0.0)
    end
end

# ── Formatting helpers ──────────────────────────────────────

function _sb_fmt_bound(probe::SBProbeResult)
    probe.is_infeasible && return "infeasible"
    isnothing(probe.dual_bound) && return "N/A"
    return @sprintf("%.4f", probe.dual_bound)
end

# ── Hook overrides ──────────────────────────────────────────

function before_branching_selection(
    lctx::BranchingLoggerContext, candidates, phases
)
    lctx.t0 = time()
    println(lctx.io, "**** Strong branching ****")
    return
end

function before_probe(
    lctx::BranchingLoggerContext, phase,
    candidate, direction::Symbol
)
    lctx.log_level < 2 && return
    label = isnothing(phase) ? "SB" : phase_label(phase)
    println(lctx.io,
        "  [$(label)] probing $(candidate.orig_var)" *
        " direction=$(direction)"
    )
    return
end

function after_probe(
    lctx::BranchingLoggerContext, phase,
    candidate, direction::Symbol, result
)
    lctx.log_level < 2 && return
    label = isnothing(phase) ? "SB" : phase_label(phase)
    bound_str = _sb_fmt_bound(result)
    println(lctx.io,
        "  [$(label)] probe $(direction)" *
        " $(candidate.orig_var): $(bound_str)"
    )
    return
end

function after_candidate_eval(
    lctx::BranchingLoggerContext, phase, idx::Int,
    candidate, score::Float64, detail
)
    label = phase_label(phase)
    et = @sprintf("%.2f", time() - lctx.t0)
    lhs = @sprintf("%.4f", candidate.value)
    sc = @sprintf("%.2f", score)

    if detail === :reliable
        println(lctx.io,
            "  [$(label)] cand. $(lpad(idx, 2))" *
            " branch on $(candidate.orig_var)" *
            " (lhs=$(lhs)): reliable," *
            " score = $(sc)  <et=$(et)>"
        )
    else
        left_str = _sb_fmt_bound(detail.left)
        right_str = _sb_fmt_bound(detail.right)
        println(lctx.io,
            "  SB cand. $(lpad(idx, 2)) branch on " *
            "$(candidate.orig_var) (lhs=$(lhs)): " *
            "[$(left_str), $(right_str)], " *
            "score = $(sc)  <et=$(et)>"
        )
    end
    return
end

function on_both_infeasible(
    lctx::BranchingLoggerContext, phase, idx::Int, candidate
)
    label = phase_label(phase)
    println(lctx.io,
        "  [$(label)] cand. $(lpad(idx, 2))" *
        " branch on $(candidate.orig_var):" *
        " both infeasible"
    )
    return
end

function after_phase_filter(
    lctx::BranchingLoggerContext, label::String,
    before::Int, after::Int
)
    println(lctx.io,
        "  [$(label)] filtered: " *
        "$(before) -> $(after) candidates"
    )
    return
end

function after_branching_selection(
    lctx::BranchingLoggerContext, candidate, score::Float64
)
    sc = @sprintf("%.2f", score)
    println(lctx.io,
        "  SB selected: $(candidate.orig_var)" *
        " (score = $(sc))"
    )
    return
end
```

- [ ] **Step 2: Make `BranchingContext` abstract**

In `src/Branching/context.jl`, change:
```julia
struct BranchingContext end
```
To:
```julia
abstract type BranchingContext end

struct DefaultBranchingContext <: BranchingContext end
```

Update all no-op stubs to dispatch on `::BranchingContext` (they already do — the abstract type will match both `DefaultBranchingContext` and `BranchingLoggerContext`).

Update `MultiPhaseStrongBranching` default in `kernel.jl`:
```julia
branching_ctx::BranchingContext = DefaultBranchingContext()
```

- [ ] **Step 3: Add include and export to `Branching.jl`**

Add `include("logger.jl")` after `include("kernel.jl")`:

```julia
include("context.jl")
include("kernel.jl")
include("logger.jl")
```

Add to exports:
```julia
export BranchingContext, DefaultBranchingContext
export BranchingLoggerContext
```

- [ ] **Step 4: Verify it compiles**

Run: `julia --project=. -e 'using Vertigo; println(Vertigo.Branching.BranchingLoggerContext())'`
Expected: prints `BranchingLoggerContext(...)` with no error.

- [ ] **Step 5: Commit**

```bash
git add src/Branching/context.jl src/Branching/logger.jl src/Branching/Branching.jl src/Branching/kernel.jl
git commit -m "add: BranchingLoggerContext with formatted hook overrides"
```

---

### Task 6: Wire `BranchingLoggerContext` into `BPSpace`

**Files:**
- Modify: `src/BranchCutPrice/space.jl`

- [ ] **Step 1: Construct logger context in `run_branch_and_price`**

In `run_branch_and_price` (lines 328-366), construct the branching context from `log_level` and pass it to the strategy. Replace the `BPSpace(...)` call:

```julia
    branching_ctx = if log_level > 0
        BranchingLoggerContext(; log_level=log_level)
    else
        DefaultBranchingContext()
    end

    # If strategy is MultiPhaseStrongBranching, override its context
    effective_strategy = if branching_strategy isa MultiPhaseStrongBranching
        MultiPhaseStrongBranching(;
            max_candidates=branching_strategy.max_candidates,
            mu=branching_strategy.mu,
            phases=branching_strategy.phases,
            reliability_threshold=branching_strategy.pseudocosts.reliability_threshold,
            branching_ctx=branching_ctx
        )
    else
        branching_strategy
    end

    space = BPSpace(
        ctx;
        node_limit = node_limit,
        tol = tol,
        rmp_time_limit = rmp_time_limit,
        rmp_heuristic = rmp_heuristic,
        separator = separator,
        max_cut_rounds = max_cut_rounds,
        min_gap_improvement = min_gap_improvement,
        branching_strategy = effective_strategy,
        log_level = log_level
    )
```

- [ ] **Step 2: Add import in `BranchCutPrice.jl`**

Ensure `BranchCutPrice` has access to `BranchingLoggerContext` and `DefaultBranchingContext` via `using ..Branching`.

- [ ] **Step 3: Verify it compiles**

Run: `julia --project=. -e 'using Vertigo'`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add src/BranchCutPrice/space.jl src/BranchCutPrice/BranchCutPrice.jl
git commit -m "update: wire BranchingLoggerContext into BPSpace"
```

---

### Task 7: Fix existing tests

**Files:**
- Modify: `test/VertigoUnitTests/branching/test_strong_branching.jl`
- Modify: `test/VertigoE2eTests/test_strong_branching_e2e.jl`

- [ ] **Step 1: Update unit test imports**

In `test/VertigoUnitTests/branching/test_strong_branching.jl`, add `BranchingContext, DefaultBranchingContext` to the import list (line 5):

```julia
using Vertigo.Branching: SBProbeResult, SBCandidateResult,
    sb_score, BranchingCandidate, find_fractional_variables,
    bp_master_model, bp_pool, bp_decomp, bp_branching_constraints,
    build_branching_terms, add_branching_constraint!,
    remove_branching_constraint!,
    bp_ip_incumbent, bp_ip_primal_bound, run_sb_probe,
    MultiPhaseStrongBranching, CGProbePhase, select_branching_variable,
    BranchingResult, branching_ok,
    DefaultBranchingContext
```

- [ ] **Step 2: Update `test_run_sb_probe_returns_dual_bounds`**

`run_sb_probe` now takes a `bctx` as first argument. Update the call (line 117):

```julia
        result = run_sb_probe(
            DefaultBranchingContext(), space, candidate,
            10, parent_lp
        )
```

- [ ] **Step 3: Update `test_run_sb_probe_restores_state`**

Same change (line 149):

```julia
        run_sb_probe(
            DefaultBranchingContext(), space, candidate,
            10, parent_lp
        )
```

- [ ] **Step 4: Run unit tests**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoUnitTests"); using VertigoUnitTests; VertigoUnitTests.run()'`
Expected: all tests pass.

- [ ] **Step 5: Run e2e tests**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoE2eTests"); using VertigoE2eTests; VertigoE2eTests.run()'`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add test/VertigoUnitTests/branching/test_strong_branching.jl test/VertigoE2eTests/test_strong_branching_e2e.jl
git commit -m "test: update branching tests for BranchingContext API"
```

---

### Task 8: Add unit tests for `BranchingLoggerContext` hooks

**Files:**
- Create: `test/VertigoUnitTests/branching/test_branching_logger.jl`
- Modify: `test/VertigoUnitTests/VertigoUnitTests.jl` (add include + call)

- [ ] **Step 1: Write tests**

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.Branching: BranchingLoggerContext, BranchingCandidate,
    SBProbeResult, SBCandidateResult, LPProbePhase, CGProbePhase,
    before_branching_selection, after_candidate_eval,
    on_both_infeasible, after_phase_filter,
    after_branching_selection

function test_logger_before_branching_selection_prints_header()
    @testset "[branching_logger] before_branching_selection prints header" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        candidates = BranchingCandidate[]
        phases = [LPProbePhase()]
        before_branching_selection(lctx, candidates, phases)
        output = String(take!(buf))
        @test contains(output, "Strong branching")
        @test lctx.t0 > 0.0
    end
end

function test_logger_after_candidate_eval_reliable()
    @testset "[branching_logger] after_candidate_eval reliable skip" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        lctx.t0 = time()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        phase = CGProbePhase(max_cg_iterations=5)
        after_candidate_eval(lctx, phase, 1, c, 4.5, :reliable)
        output = String(take!(buf))
        @test contains(output, "reliable")
        @test contains(output, "4.50")
    end
end

function test_logger_after_candidate_eval_probed()
    @testset "[branching_logger] after_candidate_eval with probe result" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        lctx.t0 = time()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        phase = LPProbePhase()
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        after_candidate_eval(lctx, phase, 1, c, 3.2, result)
        output = String(take!(buf))
        @test contains(output, "12.0000")
        @test contains(output, "14.0000")
        @test contains(output, "3.20")
    end
end

function test_logger_on_both_infeasible()
    @testset "[branching_logger] on_both_infeasible" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        phase = LPProbePhase()
        on_both_infeasible(lctx, phase, 1, c)
        output = String(take!(buf))
        @test contains(output, "both infeasible")
    end
end

function test_logger_after_phase_filter()
    @testset "[branching_logger] after_phase_filter" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        after_phase_filter(lctx, "LP", 10, 3)
        output = String(take!(buf))
        @test contains(output, "LP")
        @test contains(output, "10 -> 3")
    end
end

function test_logger_after_branching_selection()
    @testset "[branching_logger] after_branching_selection" begin
        buf = IOBuffer()
        lctx = BranchingLoggerContext(; io=buf, log_level=1)
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        after_branching_selection(lctx, c, 7.5)
        output = String(take!(buf))
        @test contains(output, "SB selected")
        @test contains(output, "7.50")
    end
end

function test_branching_logger()
    test_logger_before_branching_selection_prints_header()
    test_logger_after_candidate_eval_reliable()
    test_logger_after_candidate_eval_probed()
    test_logger_on_both_infeasible()
    test_logger_after_phase_filter()
    test_logger_after_branching_selection()
end
```

- [ ] **Step 2: Include and call in `VertigoUnitTests.jl`**

Add `include("branching/test_branching_logger.jl")` and call `test_branching_logger()` from the `run()` function.

- [ ] **Step 3: Run unit tests**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoUnitTests"); using VertigoUnitTests; VertigoUnitTests.run()'`
Expected: all tests pass (including the new logger tests).

- [ ] **Step 4: Commit**

```bash
git add test/VertigoUnitTests/branching/test_branching_logger.jl test/VertigoUnitTests/VertigoUnitTests.jl
git commit -m "test: add BranchingLoggerContext hook tests"
```

---

### Task 9: Run full test suite

**Files:** none (verification only)

- [ ] **Step 1: Run all tests**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: all unit + e2e tests pass.

- [ ] **Step 2: Verify logging output with e2e**

Run the strong branching e2e test with `log_level=2` and visually confirm output format matches the old format.

- [ ] **Step 3: Final commit if any fixups needed**

```bash
git add -u
git commit -m "fix: test suite fixups for branching logger refactor"
```
