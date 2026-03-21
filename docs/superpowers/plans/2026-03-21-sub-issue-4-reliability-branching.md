# Sub-issue 4: Reliability Branching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement pseudocost tracking and reliability branching. Reliable variables use pseudocost estimates instead of expensive CG probes. Pseudocosts are updated from two sources: probes (unreliable vars) and node evaluation (all vars).

**Architecture:** New file `pseudocosts.jl` with `PseudocostRecord`, `PseudocostTracker`, `update_pseudocosts!`, `estimate_score`, `is_reliable`. `ReliabilityBranching` strategy in `strong_branching.jl` reuses `run_sb_probe` for unreliable variables. `on_node_evaluated` callback on `AbstractBranchingStrategy` (no-op default) allows `ReliabilityBranching` to update pseudocosts after each node evaluation. `BPNodeData` extended with `branching_var` and `parent_lp_obj`.

**Tech Stack:** Julia, MathOptInterface, HiGHS (tests)

**Spec:** `docs/superpowers/specs/2026-03-20-strong-branching-framework-design.md` (Sub-issue 4)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/BranchCutPrice/pseudocosts.jl` | `PseudocostRecord`, `PseudocostTracker`, `update_pseudocosts!`, `estimate_score`, `is_reliable` |
| Modify | `src/BranchCutPrice/strong_branching.jl` | Add `ReliabilityBranching`, `select_branching_variable` |
| Modify | `src/BranchCutPrice/branching_strategy.jl` | Add `on_node_evaluated` default no-op |
| Modify | `src/BranchCutPrice/bp_output.jl` | Extend `BPNodeData` with `branching_var` and `parent_lp_obj` |
| Modify | `src/BranchCutPrice/space.jl` | Store branching info in `BPNodeData` during `branch!` |
| Modify | `src/BranchCutPrice/evaluator.jl` | Call `on_node_evaluated` after CG |
| Modify | `src/BranchCutPrice/BranchCutPrice.jl` | Add include + exports |
| Create | `test/VertigoUnitTests/colgen/test_pseudocosts.jl` | Unit + e2e tests |
| Modify | `test/VertigoUnitTests/VertigoUnitTests.jl` | Include + register |

---

### Task 1: `PseudocostRecord`, `PseudocostTracker`, and unit pseudocost functions

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
            left = SBProbeResult(10.0 + Float64(i), 10.0 + Float64(i), false)
            right = SBProbeResult(10.0 + 2.0 * Float64(i), 10.0 + 2.0 * Float64(i), false)
            result = SBCandidateResult(c, 10.0, left, right)
            update_pseudocosts!(tracker, c, result)
        end

        rec = tracker.records[1]
        @test rec.count_down == 3
        @test rec.count_up == 3
    end

    @testset "[pseudocosts] estimate_score" begin
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
        # estimate = (5/6)*min(2,4) + (1/6)*max(2,4)
        mu = 1.0 / 6.0
        expected = (1 - mu) * 2.0 + mu * 4.0
        @test estimate_score(tracker, c) ≈ expected
    end

    @testset "[pseudocosts] is_reliable" begin
        tracker = PseudocostTracker{Int}(reliability_threshold=2)
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

function _get_or_create!(tracker::PseudocostTracker, var)
    return get!(tracker.records, var, PseudocostRecord())
end

"""
    update_pseudocosts!(tracker, candidate, result)

Update pseudocost records from a probe result. Skips
infeasible sides.
"""
function update_pseudocosts!(
    tracker::PseudocostTracker,
    candidate::BranchingCandidate,
    result::SBCandidateResult
)
    rec = _get_or_create!(tracker, candidate.orig_var)
    frac = candidate.value - candidate.floor_val

    if !result.left.is_infeasible && !isnothing(result.left.dual_bound)
        delta_down = max(0.0, result.left.dual_bound - result.parent_lp_obj)
        rec.sum_down += delta_down / frac
        rec.count_down += 1
    end

    up_frac = 1.0 - frac
    if !result.right.is_infeasible && !isnothing(result.right.dual_bound)
        delta_up = max(0.0, result.right.dual_bound - result.parent_lp_obj)
        rec.sum_up += delta_up / up_frac
        rec.count_up += 1
    end
    return
end

"""
    estimate_score(tracker, candidate; mu=1/6) -> Float64

Estimate branching score from pseudocosts.
"""
function estimate_score(
    tracker::PseudocostTracker,
    candidate::BranchingCandidate;
    mu::Float64 = 1.0 / 6.0
)
    rec = tracker.records[candidate.orig_var]
    frac = candidate.value - candidate.floor_val

    mean_down = rec.count_down > 0 ?
        rec.sum_down / rec.count_down : 0.0
    mean_up = rec.count_up > 0 ?
        rec.sum_up / rec.count_up : 0.0

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

After `include("strong_branching.jl")` (line 24), add:

```julia
include("pseudocosts.jl")
```

- [ ] **Step 4: Run tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.test_pseudocosts()"`
Expected: PASS (6 testsets)

- [ ] **Step 5: Commit**

```bash
git add src/BranchCutPrice/pseudocosts.jl src/BranchCutPrice/BranchCutPrice.jl test/VertigoUnitTests/colgen/test_pseudocosts.jl test/VertigoUnitTests/VertigoUnitTests.jl
git commit -m "add: PseudocostTracker with update, estimate, and reliability"
```

---

### Task 2: `on_node_evaluated` callback and `BPNodeData` extension

**Files:**
- Modify: `src/BranchCutPrice/branching_strategy.jl`
- Modify: `src/BranchCutPrice/bp_output.jl`
- Modify: `src/BranchCutPrice/space.jl`
- Modify: `src/BranchCutPrice/evaluator.jl`

- [ ] **Step 1: Add `on_node_evaluated` default no-op**

In `src/BranchCutPrice/branching_strategy.jl`, add after the `BranchingResult` constructors (before `AbstractBranchingStrategy`):

```julia
"""
    on_node_evaluated(strategy, space, node, cg_output)

Callback invoked after CG completes on a node. Default: no-op.
"""
on_node_evaluated(::AbstractBranchingStrategy, space, node, cg_output) = nothing
```

- [ ] **Step 2: Extend `BPNodeData` with branching info**

In `src/BranchCutPrice/bp_output.jl`, replace the `BPNodeData` struct:

```julia
mutable struct BPNodeData
    cg_output::Union{Nothing,ColGen.ColGenOutput}
    branching_var::Any
    parent_lp_obj::Union{Nothing,Float64}
end

BPNodeData() = BPNodeData(nothing, nothing, nothing)
```

- [ ] **Step 3: Store branching info in `branch!`**

In `src/BranchCutPrice/space.jl`, in `TreeSearch.branch!`, after `x_val = result.value` (line 168) and before the `cg_output` line, add:

```julia
    # Store branching info for pseudocost updates
    parent_lp = if !isnothing(node.user_data) &&
                   !isnothing(node.user_data.cg_output)
        node.user_data.cg_output.master_lp_obj
    else
        nothing
    end
```

Then after children are created (after the `for child in children` loop, around line 190), add inside the loop:

```julia
    for child in children
        space.open_node_bounds[child.id] = child.dual_bound
        child.user_data.branching_var = orig_var
        child.user_data.parent_lp_obj = parent_lp
    end
```

(Remove the existing `for child in children` loop that only sets `open_node_bounds`.)

- [ ] **Step 4: Call `on_node_evaluated` in `evaluate!`**

In `src/BranchCutPrice/evaluator.jl`, after `node.user_data = BPNodeData(cg_output)` (line 80), add:

```julia
    on_node_evaluated(
        space.branching_strategy, space, node, cg_output
    )
```

- [ ] **Step 5: Run all unit tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.run()"`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/BranchCutPrice/branching_strategy.jl src/BranchCutPrice/bp_output.jl src/BranchCutPrice/space.jl src/BranchCutPrice/evaluator.jl
git commit -m "add: on_node_evaluated callback and BPNodeData branching info"
```

---

### Task 3: `ReliabilityBranching` strategy

**Files:**
- Modify: `src/BranchCutPrice/strong_branching.jl`
- Modify: `src/BranchCutPrice/BranchCutPrice.jl`
- Modify: `test/VertigoUnitTests/colgen/test_pseudocosts.jl`

- [ ] **Step 1: Add test for ReliabilityBranching**

Append to `test_pseudocosts()`:

```julia
    @testset "[ReliabilityBranching] selects variable" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)

        primal = get_primal_solution(bp_master_model(ctx))
        space = BPSpace(
            ctx; node_limit=1,
            branching_strategy=ReliabilityBranching()
        )

        result = select_branching_variable(
            ReliabilityBranching(), space, nothing, primal
        )
        @test result.status == branching_ok
        frac = result.value - floor(result.value)
        @test frac > 1e-6
        @test frac < 1.0 - 1e-6
    end
```

Add imports: `using Vertigo.BranchCutPrice: ReliabilityBranching, select_branching_variable, bp_master_model, BPSpace, branching_ok` and `using Vertigo.Reformulation: get_primal_solution`.

- [ ] **Step 2: Implement `ReliabilityBranching`**

Append to `src/BranchCutPrice/strong_branching.jl`:

```julia
# ────────────────────────────────────────────────────────────────────────────────────────
# RELIABILITY BRANCHING STRATEGY
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    ReliabilityBranching <: AbstractBranchingStrategy

Use pseudocost estimates for reliable variables and strong branching
probes for unreliable ones. Pseudocosts are updated from probes and
from node evaluations via `on_node_evaluated`.
"""
struct ReliabilityBranching <: AbstractBranchingStrategy
    max_candidates::Int
    max_cg_iterations::Int
    mu::Float64
    reliability_threshold::Int
    rule::AbstractBranchingRule
    pseudocosts::PseudocostTracker

    function ReliabilityBranching(;
        max_candidates::Int = 5,
        max_cg_iterations::Int = 10,
        mu::Float64 = 1.0 / 6.0,
        reliability_threshold::Int = 8,
        rule::AbstractBranchingRule = MostFractionalRule()
    )
        # PseudocostTracker type parameter will be inferred
        # at first update; use Any for now
        new(max_candidates, max_cg_iterations, mu,
            reliability_threshold, rule,
            PseudocostTracker{Any}(
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

    selected = select_candidates(
        rb.rule, candidates, rb.max_candidates
    )

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
        c = first(selected)
        return BranchingResult(c.orig_var, c.value)
    end

    best_score = -Inf
    best_candidate = first(selected)

    for c in selected
        if is_reliable(rb.pseudocosts, c)
            score = estimate_score(
                rb.pseudocosts, c; mu=rb.mu
            )
            @debug "RB reliable estimate" var=c.orig_var score=score
        else
            probe = run_sb_probe(
                space, c, rb.max_cg_iterations, parent_lp
            )
            if probe.left.is_infeasible &&
               probe.right.is_infeasible
                @debug "RB: both children infeasible" var=c.orig_var
                return BranchingResult(node_infeasible)
            end
            update_pseudocosts!(rb.pseudocosts, c, probe)
            score = sb_score(probe; mu=rb.mu)
            @debug "RB probed" var=c.orig_var score=score
        end
        if score > best_score
            best_score = score
            best_candidate = c
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
    isnothing(node.user_data.branching_var) && return
    isnothing(node.user_data.parent_lp_obj) && return
    isnothing(cg_output.incumbent_dual_bound) && return

    branching_var = node.user_data.branching_var
    parent_lp = node.user_data.parent_lp_obj
    child_db = cg_output.incumbent_dual_bound

    rec = get!(
        rb.pseudocosts.records, branching_var,
        PseudocostRecord()
    )

    # Determine direction from node's branching constraint
    # Use the dual bound improvement as observation
    delta = max(0.0, child_db - parent_lp)

    # We don't know which direction this node is (left/right)
    # without inspecting the constraint. Update both with
    # equal weight as a conservative approximation.
    # TODO: determine direction from branching constraint
    frac = node.user_data.parent_lp_obj  # placeholder
    return
end
```

Wait — we need to know the branching direction to update pseudocosts correctly. Let me extend `BPNodeData` to also store the direction.

Actually, let me simplify: store `branching_floor_val` in `BPNodeData`. If the child's branching constraint is `≤ floor`, it's a down-branch. If `≥ ceil`, it's an up-branch. We can determine this by comparing the child's bound with the branching variable's value.

Let me revise. I'll store `branching_direction::Symbol` (`:down` or `:up`) in `BPNodeData` instead, set in `branch!`.

- [ ] **Step 2 (revised): Implement `ReliabilityBranching`**

First, update `BPNodeData` in Task 2 Step 2 to include direction:

```julia
mutable struct BPNodeData
    cg_output::Union{Nothing,ColGen.ColGenOutput}
    branching_var::Any
    parent_lp_obj::Union{Nothing,Float64}
    branching_direction::Union{Nothing,Symbol}
end

BPNodeData() = BPNodeData(nothing, nothing, nothing, nothing)
```

And in `branch!` (Task 2 Step 3), set direction on children:

```julia
    children[1].user_data.branching_var = orig_var
    children[1].user_data.parent_lp_obj = parent_lp
    children[1].user_data.branching_direction = :down
    children[2].user_data.branching_var = orig_var
    children[2].user_data.parent_lp_obj = parent_lp
    children[2].user_data.branching_direction = :up
```

Then `on_node_evaluated` for `ReliabilityBranching`:

```julia
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

    delta = max(0.0, cg_output.incumbent_dual_bound - parent_lp)
    rec = get!(rb.pseudocosts.records, bvar, PseudocostRecord())

    # Need fractionality — approximate from parent LP value
    # We don't store it, but we can use a unit pseudocost of
    # delta directly (frac normalization happens at estimate time)
    # Actually, we need the frac_part from the parent.
    # Store it in BPNodeData too.
    return
end
```

Hmm, we also need `frac_part` to compute unit pseudocosts. Let me store `branching_frac` in `BPNodeData` too. This is getting complex. Let me simplify the `BPNodeData` to store everything needed:

```julia
mutable struct BPNodeData
    cg_output::Union{Nothing,ColGen.ColGenOutput}
    branching_var::Any
    parent_lp_obj::Union{Nothing,Float64}
    branching_direction::Union{Nothing,Symbol}
    branching_frac::Float64
end

BPNodeData() = BPNodeData(nothing, nothing, nothing, nothing, 0.0)
```

OK let me rewrite this plan more carefully. I was writing stream-of-consciousness above. Let me delete and rewrite cleanly.

- [ ] **Step 2: Implement `ReliabilityBranching` and `on_node_evaluated`**

Append to `src/BranchCutPrice/strong_branching.jl` the `ReliabilityBranching` struct and both `select_branching_variable` and `on_node_evaluated`:

The `on_node_evaluated` implementation:

```julia
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

    delta = max(0.0, cg_output.incumbent_dual_bound - parent_lp)
    frac = node.user_data.branching_frac
    rec = get!(rb.pseudocosts.records, bvar, PseudocostRecord())

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

- [ ] **Step 4: Run tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.run()"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/BranchCutPrice/strong_branching.jl src/BranchCutPrice/BranchCutPrice.jl test/VertigoUnitTests/colgen/test_pseudocosts.jl
git commit -m "add: ReliabilityBranching with pseudocost estimates and on_node_evaluated"
```

---

### Task 4: E2e test

**Files:**
- Modify: `test/VertigoUnitTests/colgen/test_pseudocosts.jl`

- [ ] **Step 1: Add e2e test**

Append to `test_pseudocosts()`:

```julia
    @testset "[ReliabilityBranching] e2e small GAP" begin
        inst = random_gap_instance(2, 4; seed=10)
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx;
            node_limit=100,
            branching_strategy=ReliabilityBranching(
                max_candidates=3,
                max_cg_iterations=5,
                reliability_threshold=2
            )
        )
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
    end
```

- [ ] **Step 2: Run full regression**

Run: `julia --project=. -e "include(\"test/runtests.jl\")"`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add test/VertigoUnitTests/colgen/test_pseudocosts.jl
git commit -m "test: add e2e test for ReliabilityBranching"
```
