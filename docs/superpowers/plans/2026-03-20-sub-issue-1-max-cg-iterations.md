# Sub-issue 1: Configurable `max_cg_iterations` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CG iteration limit configurable on `ColGenContext` instead of hardcoded to 1000.

**Architecture:** Add a `max_cg_iterations` field to `ColGenContext`, replace the hardcoded `1000` in `stop_colgen_phase`, and forward accessors through `ColGenLoggerContext`. The limit applies per phase (Phase 0 and Phase 2). Phase 1 remains unlimited.

**Tech Stack:** Julia, MathOptInterface, HiGHS (tests)

**Spec:** `docs/superpowers/specs/2026-03-20-strong-branching-framework-design.md` (Sub-issue 1)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/ColGen/context.jl` | Add field, constructor kwarg, accessor, mutator, use in `stop_colgen_phase` |
| Modify | `src/ColGen/logger.jl` | Forward accessor/mutator to `lctx.inner` |
| Create | `test/VertigoUnitTests/colgen/test_max_cg_iterations.jl` | Unit tests for the new field and stop logic |
| Modify | `test/VertigoUnitTests/VertigoUnitTests.jl` | Include + register new test |

---

### Task 1: Write failing test for `max_cg_iterations` accessor and default

**Files:**
- Create: `test/VertigoUnitTests/colgen/test_max_cg_iterations.jl`
- Modify: `test/VertigoUnitTests/VertigoUnitTests.jl`

- [ ] **Step 1: Write the test file**

Create `test/VertigoUnitTests/colgen/test_max_cg_iterations.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.ColGen: ColGenContext, max_cg_iterations,
    set_max_cg_iterations!

function test_max_cg_iterations()
    @testset "[max_cg_iterations] default value" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst)
        @test max_cg_iterations(ctx) == 1000
    end

    @testset "[max_cg_iterations] set and get" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst)
        set_max_cg_iterations!(ctx, 10)
        @test max_cg_iterations(ctx) == 10
    end

    @testset "[max_cg_iterations] constructor kwarg" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst; max_cg_iterations=50)
        @test max_cg_iterations(ctx) == 50
    end
end
```

- [ ] **Step 2: Register the test in VertigoUnitTests.jl**

In `test/VertigoUnitTests/VertigoUnitTests.jl`:
- Add `include("colgen/test_max_cg_iterations.jl")` after line 149 (after `test_cut_pool_tracker.jl` include).
- Add `test_max_cg_iterations()` in the `run()` function after `test_cut_pool_tracker()`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoUnitTests"); using VertigoUnitTests; VertigoUnitTests.test_max_cg_iterations()'`
Expected: FAIL — `max_cg_iterations` is not exported / does not exist.

---

### Task 2: Add `max_cg_iterations` field and accessor to `ColGenContext`

**Files:**
- Modify: `src/ColGen/context.jl:112-136`

- [ ] **Step 1: Add the field to the struct**

In `src/ColGen/context.jl`, add `max_cg_iterations::Int` after `smoothing_alpha::Float64` (line 123):

```julia
mutable struct ColGenContext{D<:AbstractDecomposition}
    decomp::D
    pool::ColumnPool
    # TODO: support non-robust cuts
    eq_art_vars::Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}
    leq_art_vars::Dict{TaggedCI,MOI.VariableIndex}
    geq_art_vars::Dict{TaggedCI,MOI.VariableIndex}
    ip_incumbent::Union{Nothing,MasterIpPrimalSol}
    ip_primal_bound::Union{Nothing,Float64}
    branching_constraints::Vector{ActiveBranchingConstraint}
    robust_cuts::Vector{ActiveRobustCut}
    smoothing_alpha::Float64
    max_cg_iterations::Int
```

- [ ] **Step 2: Update the constructor**

Replace the inner constructor (lines 125-136) to accept the new kwarg with default 1000:

```julia
    function ColGenContext(
        decomp, pool,
        eq_art_vars, leq_art_vars, geq_art_vars;
        smoothing_alpha::Float64 = 0.0,
        max_cg_iterations::Int = 1000
    )
        new{typeof(decomp)}(
            decomp, pool,
            eq_art_vars, leq_art_vars, geq_art_vars, nothing,
            nothing, ActiveBranchingConstraint[],
            ActiveRobustCut[], smoothing_alpha, max_cg_iterations
        )
    end
```

- [ ] **Step 3: Add accessor and mutator**

Add after `is_minimization(ctx::ColGenContext)` (after line 140):

```julia
max_cg_iterations(ctx::ColGenContext) = ctx.max_cg_iterations

function set_max_cg_iterations!(ctx::ColGenContext, n::Int)
    ctx.max_cg_iterations = n
    return
end
```

- [ ] **Step 4: Update `build_gap_context` in test helper to forward kwarg**

In `test/VertigoUnitTests/VertigoUnitTests.jl`, update `build_gap_context` (line 56) signature and constructor call:

```julia
function build_gap_context(inst::GAPInstance; max_cg_iterations::Int=1000)
```

And update the `ColGenContext` call (line 123-129):

```julia
    ctx = ColGenContext(
        decomp,
        pool,
        Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}(),
        Dict{TaggedCI,MOI.VariableIndex}(),
        Dict{TaggedCI,MOI.VariableIndex}();
        max_cg_iterations=max_cg_iterations
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoUnitTests"); using VertigoUnitTests; VertigoUnitTests.test_max_cg_iterations()'`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/ColGen/context.jl test/VertigoUnitTests/VertigoUnitTests.jl test/VertigoUnitTests/colgen/test_max_cg_iterations.jl
git commit -m "add: max_cg_iterations field and accessor on ColGenContext"
```

---

### Task 3: Write failing test for `stop_colgen_phase` respecting the limit

**Files:**
- Modify: `test/VertigoUnitTests/colgen/test_max_cg_iterations.jl`

- [ ] **Step 1: Add import for stop_colgen_phase and Phase types**

At the top of `test_max_cg_iterations.jl`, update the imports:

```julia
using Vertigo.ColGen: ColGenContext, max_cg_iterations,
    set_max_cg_iterations!, stop_colgen_phase,
    ColGenIterationOutput, Phase0, Phase1, Phase2
```

- [ ] **Step 2: Add test for stop_colgen_phase with custom limit**

Append to the `test_max_cg_iterations()` function:

```julia
    @testset "[max_cg_iterations] stop_colgen_phase respects limit" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst; max_cg_iterations=5)

        # Fake iteration output: columns added, no gap closed
        iter_output = ColGenIterationOutput(
            100.0,   # master_lp_obj
            nothing, # dual_bound
            1,       # nb_columns_added (not zero)
            nothing, # master_lp_dual_sol
            nothing, # master_ip_primal_sol
            false    # subproblem_infeasible
        )

        # Phase0: iteration 4 (≤ 5) → don't stop
        @test !stop_colgen_phase(ctx, Phase0(), iter_output, nothing, nothing, 4)
        # Phase0: iteration 6 (> 5) → stop
        @test stop_colgen_phase(ctx, Phase0(), iter_output, nothing, nothing, 6)
        # Phase2: same behavior
        @test !stop_colgen_phase(ctx, Phase2(), iter_output, nothing, nothing, 5)
        @test stop_colgen_phase(ctx, Phase2(), iter_output, nothing, nothing, 6)
    end

    @testset "[max_cg_iterations] Phase1 ignores limit" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst; max_cg_iterations=5)

        # Phase1 iter output: columns still being added
        iter_output = ColGenIterationOutput(
            100.0, nothing, 1, nothing, nothing, false
        )

        # Phase1 does not stop even beyond limit
        @test !stop_colgen_phase(ctx, Phase1(), iter_output, nothing, nothing, 100)
    end
```

- [ ] **Step 3: Run test to verify Phase0/Phase2 tests fail**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoUnitTests"); using VertigoUnitTests; VertigoUnitTests.test_max_cg_iterations()'`
Expected: FAIL — `stop_colgen_phase` still uses hardcoded 1000, so iteration 6 won't trigger a stop.

---

### Task 4: Replace hardcoded 1000 in `stop_colgen_phase`

**Files:**
- Modify: `src/ColGen/context.jl:376-396`

- [ ] **Step 1: Replace the hardcoded limit**

In `stop_colgen_phase` for `Union{Phase0,Phase2}` (line 386), change:

```julia
    iteration_limit = iteration > 1000
```

to:

```julia
    iteration_limit = iteration > max_cg_iterations(ctx)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoUnitTests"); using VertigoUnitTests; VertigoUnitTests.test_max_cg_iterations()'`
Expected: PASS (all tests including the new stop_colgen_phase tests)

- [ ] **Step 3: Run all unit tests for regression**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoUnitTests"); using VertigoUnitTests; VertigoUnitTests.run()'`
Expected: All tests PASS (default 1000 preserves existing behavior)

- [ ] **Step 4: Commit**

```bash
git add src/ColGen/context.jl test/VertigoUnitTests/colgen/test_max_cg_iterations.jl
git commit -m "refactor: use configurable max_cg_iterations in stop_colgen_phase"
```

---

### Task 5: Forward through `ColGenLoggerContext`

**Files:**
- Modify: `src/ColGen/logger.jl:38-65`
- Modify: `test/VertigoUnitTests/colgen/test_max_cg_iterations.jl`

- [ ] **Step 1: Write failing test for logger context forwarding**

Add to `test_max_cg_iterations.jl` imports:

```julia
using Vertigo.ColGen: ColGenLoggerContext
```

Append to the `test_max_cg_iterations()` function:

```julia
    @testset "[max_cg_iterations] ColGenLoggerContext forwarding" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst; max_cg_iterations=42)
        lctx = ColGenLoggerContext(ctx; log_level=0)
        @test max_cg_iterations(lctx) == 42
        set_max_cg_iterations!(lctx, 7)
        @test max_cg_iterations(lctx) == 7
        # Verify it changed the inner context too
        @test max_cg_iterations(ctx) == 7
    end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoUnitTests"); using VertigoUnitTests; VertigoUnitTests.test_max_cg_iterations()'`
Expected: FAIL — `max_cg_iterations(::ColGenLoggerContext)` not defined.

- [ ] **Step 3: Add forwarding methods to logger.jl**

In `src/ColGen/logger.jl`, add in the "Protocol delegation (context as arg 1)" section (after line 64):

```julia
max_cg_iterations(lctx::ColGenLoggerContext)                        = max_cg_iterations(lctx.inner)
set_max_cg_iterations!(lctx::ColGenLoggerContext, n::Int)           = set_max_cg_iterations!(lctx.inner, n)
```

- [ ] **Step 4: Run all tests**

Run: `julia --project=. -e 'push!(LOAD_PATH, "test/VertigoUnitTests"); using VertigoUnitTests; VertigoUnitTests.run()'`
Expected: All PASS

- [ ] **Step 5: Run e2e tests for full regression**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/ColGen/logger.jl test/VertigoUnitTests/colgen/test_max_cg_iterations.jl
git commit -m "add: forward max_cg_iterations through ColGenLoggerContext"
```
