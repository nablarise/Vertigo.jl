# Sub-issue 4: Reliability Branching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement pseudocost tracking and reliability branching (Algorithm 3 of Achterberg et al., 2005). Reliable variables use pseudocost estimates; unreliable variables get CG probes. Lookahead λ stops probing early when the best score stabilizes. Candidates are sorted by pseudocost score before probing.

**Architecture:** New file `pseudocosts.jl` with `PseudocostRecord`, `PseudocostTracker{X}` (parameterized on variable type), `update_pseudocosts!`, `estimate_score` (with global average fallback), `is_reliable`. `ReliabilityBranching` strategy in `strong_branching.jl` with lookahead λ and pseudocost-sorted candidate ordering. `on_node_evaluated` callback on `AbstractBranchingStrategy` (no-op default) for pseudocost updates after each node. `BPNodeData` extended with 4 fields: `branching_var`, `parent_lp_obj`, `branching_direction`, `branching_frac`.

**Tech Stack:** Julia, MathOptInterface, HiGHS (tests)

**Spec:** `docs/superpowers/specs/2026-03-20-strong-branching-framework-design.md` (Sub-issue 4)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/BranchCutPrice/pseudocosts.jl` | `PseudocostRecord`, `PseudocostTracker{X}`, `update_pseudocosts!`, `estimate_score`, `is_reliable`, `global_average_pseudocost` |
| Modify | `src/BranchCutPrice/strong_branching.jl` | Add `ReliabilityBranching`, `select_branching_variable`, `on_node_evaluated` |
| Modify | `src/BranchCutPrice/branching_strategy.jl` | Add `on_node_evaluated` default no-op |
| Modify | `src/BranchCutPrice/bp_output.jl` | Extend `BPNodeData` with `branching_var`, `parent_lp_obj`, `branching_direction`, `branching_frac` |
| Modify | `src/BranchCutPrice/space.jl` | Store branching info in children during `branch!` |
| Modify | `src/BranchCutPrice/evaluator.jl` | Call `on_node_evaluated` after CG |
| Modify | `src/BranchCutPrice/BranchCutPrice.jl` | Add include + exports |
| Create | `test/VertigoUnitTests/colgen/test_pseudocosts.jl` | Unit + e2e tests |
| Modify | `test/VertigoUnitTests/VertigoUnitTests.jl` | Include + register |

---

### Task 1: Pseudocost types and functions

**Files:**
- Create: `src/BranchCutPrice/pseudocosts.jl`
- Create: `test/VertigoUnitTests/colgen/test_pseudocosts.jl`
- Modify: `test/VertigoUnitTests/VertigoUnitTests.jl`
- Modify: `src/BranchCutPrice/BranchCutPrice.jl`

- [ ] **Step 1: Write test file**

Create `test/VertigoUnitTests/colgen/test_pseudocosts.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.BranchCutPrice: PseudocostRecord, PseudocostTracker,
    update_pseudocosts!, estimate_score, is_reliable,
    global_average_pseudocost,
    BranchingCandidate, SBProbeResult, SBCandidateResult

function test_pseudocosts()
    @testset "[pseudocosts] cold start" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        @test !is_reliable(tracker, c)
        @test !haskey(tracker.records, 1)
    end

    @testset "[pseudocosts] update from probe result" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)

        update_pseudocosts!(tracker, c, result)

        rec = tracker.records[1]
        # frac_part = 0.3
        # Δ⁻ = 12.0 - 10.0 = 2.0, unit_down = 2.0 / 0.3
        @test rec.count_down == 1
        @test rec.sum_down ≈ 2.0 / 0.3
        # Δ⁺ = 14.0 - 10.0 = 4.0, unit_up = 4.0 / 0.7
        @test rec.count_up == 1
        @test rec.sum_up ≈ 4.0 / 0.7
    end

    @testset "[pseudocosts] skip infeasible side" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(nothing, nothing, true)
        result = SBCandidateResult(c, 10.0, left, right)

        update_pseudocosts!(tracker, c, result)

        rec = tracker.records[1]
        @test rec.count_down == 1
        @test rec.count_up == 0
        @test rec.sum_up == 0.0
    end

    @testset "[pseudocosts] multiple observations" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)

        for i in 1:3
            left = SBProbeResult(
                10.0 + Float64(i), 10.0 + Float64(i), false
            )
            right = SBProbeResult(
                10.0 + 2.0 * Float64(i),
                10.0 + 2.0 * Float64(i), false
            )
            result = SBCandidateResult(c, 10.0, left, right)
            update_pseudocosts!(tracker, c, result)
        end

        rec = tracker.records[1]
        @test rec.count_down == 3
        @test rec.count_up == 3
    end

    @testset "[pseudocosts] estimate_score with observations" begin
        tracker = PseudocostTracker{Int}()
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        update_pseudocosts!(tracker, c, result)

        # mean_unit_down = (2.0/0.3) / 1 = 6.667
        # mean_unit_up = (4.0/0.7) / 1 = 5.714
        # score_down = 6.667 * 0.3 = 2.0
        # score_up = 5.714 * 0.7 = 4.0
        mu = 1.0 / 6.0
        expected = (1 - mu) * 2.0 + mu * 4.0
        @test estimate_score(tracker, c) ≈ expected
    end

    @testset "[pseudocosts] estimate_score uses global average fallback" begin
        tracker = PseudocostTracker{Int}()
        # Var 1 has observations
        c1 = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c1, 10.0, left, right)
        update_pseudocosts!(tracker, c1, result)

        # Var 2 has no observations — should use global average
        c2 = BranchingCandidate(2, 3.4, 3.0, 4.0, 0.4)
        score = estimate_score(tracker, c2)
        @test score > 0.0  # not zero — uses global average
    end

    @testset "[pseudocosts] is_reliable" begin
        tracker = PseudocostTracker{Int}(
            reliability_threshold=2
        )
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        @test !is_reliable(tracker, c)

        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)

        update_pseudocosts!(tracker, c, result)
        @test !is_reliable(tracker, c)

        update_pseudocosts!(tracker, c, result)
        @test is_reliable(tracker, c)
    end

    @testset "[pseudocosts] global_average_pseudocost" begin
        tracker = PseudocostTracker{Int}()
        # Empty tracker — returns (1.0, 1.0) per Achterberg §2.2
        avg_down, avg_up = global_average_pseudocost(tracker)
        @test avg_down == 1.0
        @test avg_up == 1.0

        # Add one observation
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        update_pseudocosts!(tracker, c, result)

        avg_down, avg_up = global_average_pseudocost(tracker)
        @test avg_down ≈ 2.0 / 0.3
        @test avg_up ≈ 4.0 / 0.7
    end
end
```

Register in `test/VertigoUnitTests/VertigoUnitTests.jl`:
- Add `include("colgen/test_pseudocosts.jl")` after the `test_strong_branching.jl` include.
- Add `test_pseudocosts()` in `run()` after `test_strong_branching()`.

- [ ] **Step 2: Create `pseudocosts.jl`**

Create `src/BranchCutPrice/pseudocosts.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# PSEUDOCOST TRACKING
# ────────────────────────────────────────────────────────────────────────────────────────

mutable struct PseudocostRecord
    sum_down::Float64
    count_down::Int
    sum_up::Float64
    count_up::Int
end

PseudocostRecord() = PseudocostRecord(0.0, 0, 0.0, 0)

struct PseudocostTracker{X}
    records::Dict{X,PseudocostRecord}
    reliability_threshold::Int
end

function PseudocostTracker{X}(;
    reliability_threshold::Int = 8
) where {X}
    return PseudocostTracker{X}(
        Dict{X,PseudocostRecord}(), reliability_threshold
    )
end

"""
    update_pseudocosts!(tracker, candidate, result)

Update pseudocost records from a probe result. Skips
infeasible sides and sides with no dual bound.
"""
function update_pseudocosts!(
    tracker::PseudocostTracker,
    candidate::BranchingCandidate,
    result::SBCandidateResult
)
    rec = get!(
        tracker.records, candidate.orig_var, PseudocostRecord()
    )
    frac = candidate.value - candidate.floor_val

    if !result.left.is_infeasible &&
       !isnothing(result.left.dual_bound)
        delta = max(0.0, result.left.dual_bound - result.parent_lp_obj)
        rec.sum_down += delta / frac
        rec.count_down += 1
    end

    up_frac = 1.0 - frac
    if !result.right.is_infeasible &&
       !isnothing(result.right.dual_bound)
        delta = max(0.0, result.right.dual_bound - result.parent_lp_obj)
        rec.sum_up += delta / up_frac
        rec.count_up += 1
    end
    return
end

"""
    global_average_pseudocost(tracker) -> (avg_down, avg_up)

Compute the global average unit pseudocost across all variables
with observations. Returns `(0.0, 0.0)` if no observations exist.
"""
function global_average_pseudocost(tracker::PseudocostTracker)
    total_down = 0.0
    n_down = 0
    total_up = 0.0
    n_up = 0
    for (_, rec) in tracker.records
        if rec.count_down > 0
            total_down += rec.sum_down / rec.count_down
            n_down += 1
        end
        if rec.count_up > 0
            total_up += rec.sum_up / rec.count_up
            n_up += 1
        end
    end
    avg_down = n_down > 0 ? total_down / n_down : 1.0
    avg_up = n_up > 0 ? total_up / n_up : 1.0
    return avg_down, avg_up
end

"""
    estimate_score(tracker, candidate; mu=1/6) -> Float64

Estimate branching score from pseudocosts. Uses global average
as fallback for variables with no observations on one or both
sides (Achterberg et al., 2005, §2.2).
"""
function estimate_score(
    tracker::PseudocostTracker,
    candidate::BranchingCandidate;
    mu::Float64 = 1.0 / 6.0
)
    frac = candidate.value - candidate.floor_val
    avg_down, avg_up = global_average_pseudocost(tracker)

    if haskey(tracker.records, candidate.orig_var)
        rec = tracker.records[candidate.orig_var]
        mean_down = rec.count_down > 0 ?
            rec.sum_down / rec.count_down : avg_down
        mean_up = rec.count_up > 0 ?
            rec.sum_up / rec.count_up : avg_up
    else
        mean_down = avg_down
        mean_up = avg_up
    end

    score_down = mean_down * frac
    score_up = mean_up * (1.0 - frac)

    return (1 - mu) * min(score_down, score_up) +
           mu * max(score_down, score_up)
end

"""
    is_reliable(tracker, candidate) -> Bool

True when `min(count_down, count_up) >= reliability_threshold`.
"""
function is_reliable(
    tracker::PseudocostTracker,
    candidate::BranchingCandidate
)
    !haskey(tracker.records, candidate.orig_var) && return false
    rec = tracker.records[candidate.orig_var]
    return min(rec.count_down, rec.count_up) >=
           tracker.reliability_threshold
end
```

- [ ] **Step 3: Add include in `BranchCutPrice.jl`**

After `include("strong_branching.jl")`, add:

```julia
include("pseudocosts.jl")
```

- [ ] **Step 4: Run tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.test_pseudocosts()"`
Expected: PASS (8 testsets)

- [ ] **Step 5: Commit**

```bash
git add src/BranchCutPrice/pseudocosts.jl src/BranchCutPrice/BranchCutPrice.jl test/VertigoUnitTests/colgen/test_pseudocosts.jl test/VertigoUnitTests/VertigoUnitTests.jl
git commit -m "add: PseudocostTracker with update, estimate, and reliability"
```

---

### Task 2: `on_node_evaluated` callback, `BPNodeData` extension, wiring

**Files:**
- Modify: `src/BranchCutPrice/branching_strategy.jl`
- Modify: `src/BranchCutPrice/bp_output.jl`
- Modify: `src/BranchCutPrice/space.jl`
- Modify: `src/BranchCutPrice/evaluator.jl`

- [ ] **Step 1: Add `on_node_evaluated` default no-op to `branching_strategy.jl`**

Add after the `AbstractBranchingStrategy` definition (before `MostFractionalBranching`):

```julia
"""
    on_node_evaluated(strategy, space, node, cg_output)

Callback after CG completes on a node. Default: no-op.
"""
on_node_evaluated(::AbstractBranchingStrategy, space, node, cg_output) = nothing
```

- [ ] **Step 2: Extend `BPNodeData` in `bp_output.jl`**

Replace the `BPNodeData` struct with all 5 fields:

```julia
mutable struct BPNodeData
    cg_output::Union{Nothing,ColGen.ColGenOutput}
    branching_var::Any
    parent_lp_obj::Union{Nothing,Float64}
    branching_direction::Union{Nothing,Symbol}
    branching_frac::Union{Nothing,Float64}
end

BPNodeData() = BPNodeData(nothing, nothing, nothing, nothing, nothing)
```

- [ ] **Step 3: Store branching info in children during `branch!`**

In `src/BranchCutPrice/space.jl`, in `TreeSearch.branch!`, after `x_val = result.value` and before the `cg_output` line, compute `parent_lp`:

```julia
    parent_lp = if !isnothing(node.user_data) &&
                   !isnothing(node.user_data.cg_output)
        node.user_data.cg_output.master_lp_obj
    else
        nothing
    end
    branching_frac = x_val - floor(x_val)
```

Then replace the existing `for child in children` loop (that only sets `open_node_bounds`) with:

```julia
    children[1].user_data.branching_var = orig_var
    children[1].user_data.parent_lp_obj = parent_lp
    children[1].user_data.branching_direction = :down
    children[1].user_data.branching_frac = branching_frac
    children[2].user_data.branching_var = orig_var
    children[2].user_data.parent_lp_obj = parent_lp
    children[2].user_data.branching_direction = :up
    children[2].user_data.branching_frac = branching_frac
    for child in children
        space.open_node_bounds[child.id] = child.dual_bound
    end
```

- [ ] **Step 4: Call `on_node_evaluated` in `evaluate!`**

In `src/BranchCutPrice/evaluator.jl`, after `node.user_data = BPNodeData(cg_output)` (line 80), add:

```julia
    on_node_evaluated(
        space.branching_strategy, space, node, cg_output
    )
```

Note: `BPNodeData(cg_output)` needs a new convenience constructor. Add to `bp_output.jl`:

```julia
BPNodeData(cg_output) = BPNodeData(cg_output, nothing, nothing, nothing, nothing)
```

Wait — `evaluate!` constructs `BPNodeData(cg_output)`, which now needs to match the 5-field constructor. The existing call `BPNodeData(cg_output)` won't work with 5 fields. Add a 1-arg constructor:

```julia
BPNodeData(cg_output::ColGen.ColGenOutput) =
    BPNodeData(cg_output, nothing, nothing, nothing, nothing)
```

But `node.user_data` already has `branching_var` etc. set by `branch!` in the parent. `evaluate!` overwrites it with a fresh `BPNodeData(cg_output)`, losing the branching info! The fix: preserve existing branching fields:

```julia
    node.user_data.cg_output = cg_output
```

But first add a guard for the root node which has no `user_data`:

```julia
    if isnothing(node.user_data)
        node.user_data = BPNodeData()
    end
    node.user_data.cg_output = cg_output
```

This preserves the branching info set by `branch!` on non-root nodes, and creates a fresh `BPNodeData` for the root.

- [ ] **Step 5: Run all unit tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.run()"`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/BranchCutPrice/branching_strategy.jl src/BranchCutPrice/bp_output.jl src/BranchCutPrice/space.jl src/BranchCutPrice/evaluator.jl
git commit -m "add: on_node_evaluated callback and BPNodeData branching info"
```

---

### Task 3: `ReliabilityBranching` strategy with lookahead and pseudocost sorting

**Files:**
- Modify: `src/BranchCutPrice/strong_branching.jl`
- Modify: `src/BranchCutPrice/BranchCutPrice.jl`
- Modify: `test/VertigoUnitTests/colgen/test_pseudocosts.jl`

- [ ] **Step 1: Add tests for ReliabilityBranching**

Append to `test_pseudocosts()`. Add imports at top:

```julia
using Vertigo.BranchCutPrice: ReliabilityBranching,
    select_branching_variable, bp_master_model, BPSpace,
    branching_ok, run_sb_probe, StrongBranching
using Vertigo.Reformulation: get_primal_solution
using Vertigo.BranchCutPrice: BPNodeData
using Vertigo.ColGen: ColGenOutput, optimal
```

Tests:

```julia
    @testset "[ReliabilityBranching] selects variable with cg_output" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        cg_out = run_column_generation(ctx)

        primal = get_primal_solution(bp_master_model(ctx))
        rb = ReliabilityBranching(
            max_candidates=10, max_cg_iterations=5,
            reliability_threshold=2
        )
        space = BPSpace(
            ctx; node_limit=1, branching_strategy=rb
        )

        # Create a mock node with cg_output
        node_data = BPNodeData()
        node_data.cg_output = cg_out
        mock_node = (user_data=node_data,)

        result = select_branching_variable(
            rb, space, mock_node, primal
        )
        @test result.status == branching_ok
        frac = result.value - floor(result.value)
        @test frac > 1e-6
        @test frac < 1.0 - 1e-6
    end

    @testset "[ReliabilityBranching] lookahead stops early" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        cg_out = run_column_generation(ctx)
        primal = get_primal_solution(bp_master_model(ctx))

        # lookahead=1, all unreliable → at most 2 probed
        rb = ReliabilityBranching(
            max_candidates=100, max_cg_iterations=5,
            reliability_threshold=100, lookahead=1
        )
        space = BPSpace(
            ctx; node_limit=1, branching_strategy=rb
        )
        node_data = BPNodeData()
        node_data.cg_output = cg_out
        mock_node = (user_data=node_data,)

        result = select_branching_variable(
            rb, space, mock_node, primal
        )
        @test result.status == branching_ok

        # Verify lookahead actually cut the loop
        n_probed = length(rb.pseudocosts.records)
        @test n_probed <= 2
    end
```

- [ ] **Step 2: Implement `ReliabilityBranching`**

Append to `src/BranchCutPrice/strong_branching.jl`:

```julia
# ────────────────────────────────────────────────────────────────────────────────────────
# RELIABILITY BRANCHING STRATEGY
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    ReliabilityBranching <: AbstractBranchingStrategy

Reliability branching (Achterberg et al., 2005, Algorithm 3).
Uses pseudocost estimates for reliable variables and CG probes
for unreliable ones. Candidates sorted by pseudocost score.
Lookahead λ stops probing when the best score stabilizes.
"""
struct ReliabilityBranching <: AbstractBranchingStrategy
    max_candidates::Int
    max_cg_iterations::Int
    mu::Float64
    reliability_threshold::Int
    lookahead::Int
    pseudocosts::PseudocostTracker{MOI.VariableIndex}

    function ReliabilityBranching(;
        max_candidates::Int = 100,
        max_cg_iterations::Int = 10,
        mu::Float64 = 1.0 / 6.0,
        reliability_threshold::Int = 8,
        lookahead::Int = 8
    )
        new(max_candidates, max_cg_iterations, mu,
            reliability_threshold, lookahead,
            PseudocostTracker{MOI.VariableIndex}(
                reliability_threshold=reliability_threshold
            ))
    end
end

function select_branching_variable(
    rb::ReliabilityBranching, space, node,
    primal_values::Dict{MOI.VariableIndex,Float64}
)
    ctx = space.ctx
    candidates = find_fractional_variables(
        ctx, primal_values; tol=space.tol
    )
    isempty(candidates) && return BranchingResult(all_integral)

    parent_lp = if !isnothing(node) &&
                   !isnothing(node.user_data) &&
                   !isnothing(node.user_data.cg_output)
        node.user_data.cg_output.master_lp_obj
    else
        nothing
    end

    if isnothing(parent_lp)
        @warn "ReliabilityBranching: no parent LP obj, " *
              "falling back to most fractional candidate"
        c = first(candidates)
        return BranchingResult(c.orig_var, c.value)
    end

    # Sort ALL candidates by pseudocost score descending
    scored = [(c, estimate_score(rb.pseudocosts, c; mu=rb.mu))
              for c in candidates]
    sort!(scored; by=x -> x[2], rev=true)
    # Apply max_candidates as safety guard
    if length(scored) > rb.max_candidates
        resize!(scored, rb.max_candidates)
    end

    best_score = -Inf
    best_candidate = scored[1][1]
    no_improvement_count = 0

    for (c, pc_score) in scored
        if is_reliable(rb.pseudocosts, c)
            score = pc_score
            @debug "RB reliable" var=c.orig_var score=score
        else
            probe = run_sb_probe(
                space, c, rb.max_cg_iterations, parent_lp
            )
            if probe.left.is_infeasible &&
               probe.right.is_infeasible
                @debug "RB: both infeasible" var=c.orig_var
                return BranchingResult(node_infeasible)
            end
            update_pseudocosts!(rb.pseudocosts, c, probe)
            score = sb_score(probe; mu=rb.mu)
            @debug "RB probed" var=c.orig_var score=score
        end

        if score > best_score
            best_score = score
            best_candidate = c
            no_improvement_count = 0
        else
            no_improvement_count += 1
        end

        # Lookahead: stop if best hasn't changed for λ candidates
        if no_improvement_count >= rb.lookahead
            @debug "RB lookahead triggered after" count=no_improvement_count
            break
        end
    end

    @debug "RB selected" var=best_candidate.orig_var score=best_score
    return BranchingResult(
        best_candidate.orig_var, best_candidate.value
    )
end

function on_node_evaluated(
    rb::ReliabilityBranching, space, node, cg_output
)
    bvar = node.user_data.branching_var
    isnothing(bvar) && return
    parent_lp = node.user_data.parent_lp_obj
    isnothing(parent_lp) && return
    isnothing(cg_output.incumbent_dual_bound) && return
    dir = node.user_data.branching_direction
    isnothing(dir) && return
    frac = node.user_data.branching_frac
    isnothing(frac) && return

    delta = max(0.0, cg_output.incumbent_dual_bound - parent_lp)
    rec = get!(
        rb.pseudocosts.records, bvar, PseudocostRecord()
    )

    if dir === :down && frac > 0.0
        rec.sum_down += delta / frac
        rec.count_down += 1
    elseif dir === :up && (1.0 - frac) > 0.0
        rec.sum_up += delta / (1.0 - frac)
        rec.count_up += 1
    end
    return
end
```

- [ ] **Step 3: Add export**

In `BranchCutPrice.jl`, add `ReliabilityBranching` to the strategy export line.

- [ ] **Step 4: Run all unit tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.run()"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/BranchCutPrice/strong_branching.jl src/BranchCutPrice/BranchCutPrice.jl test/VertigoUnitTests/colgen/test_pseudocosts.jl
git commit -m "add: ReliabilityBranching with lookahead and pseudocost sorting"
```

---

### Task 4: E2e test and pseudocost verification

**Files:**
- Modify: `test/VertigoUnitTests/colgen/test_pseudocosts.jl`

- [ ] **Step 1: Add e2e test with pseudocost verification**

Append to `test_pseudocosts()`:

```julia
    @testset "[ReliabilityBranching] e2e small GAP" begin
        inst = random_gap_instance(2, 4; seed=10)
        rb = ReliabilityBranching(
            max_candidates=10, max_cg_iterations=5,
            reliability_threshold=2
        )
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx;
            node_limit=100,
            branching_strategy=rb
        )
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
        # Verify pseudocosts were actually updated
        @test !isempty(rb.pseudocosts.records)
    end
```

- [ ] **Step 2: Run full regression**

Run: `julia --project=. -e "include(\"test/runtests.jl\")"`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add test/VertigoUnitTests/colgen/test_pseudocosts.jl
git commit -m "test: add e2e test for ReliabilityBranching with pseudocost verification"
```
