# Sub-issue 2: Branching Strategy Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a pluggable branching strategy interface with `BranchingCandidate`, branching rules, and `AbstractBranchingStrategy`. Refactor `BPSpace` and `branch!` to use the strategy pattern. Default behavior (`MostFractionalBranching`) delegates to the existing `most_fractional_original_variable` — zero behavior change.

**Architecture:** Three new files (`branching_candidates.jl`, `branching_rules.jl`, `branching_strategy.jl`) included between `branching.jl` and `cut_col_gen.jl`. `BPSpace` gets a `branching_strategy` field. `branch!` calls `select_branching_variable(strategy, space, node, primal_values)` instead of `most_fractional_original_variable` directly.

**Tech Stack:** Julia, MathOptInterface, HiGHS (tests)

**Spec:** `docs/superpowers/specs/2026-03-20-strong-branching-framework-design.md` (Sub-issue 2)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/BranchCutPrice/branching_candidates.jl` | `BranchingCandidate` struct, `find_fractional_variables` |
| Create | `src/BranchCutPrice/branching_rules.jl` | `AbstractBranchingRule`, `MostFractionalRule`, `LeastFractionalRule`, `select_candidates` |
| Create | `src/BranchCutPrice/branching_strategy.jl` | `AbstractBranchingStrategy`, `MostFractionalBranching`, `select_branching_variable` |
| Modify | `src/BranchCutPrice/BranchCutPrice.jl` | Add includes + exports |
| Modify | `src/BranchCutPrice/space.jl` | Add `branching_strategy` field to `BPSpace`, kwarg to constructor and `run_branch_and_price` |
| Modify | `src/BranchCutPrice/space.jl` | Refactor `branch!` to use `select_branching_variable` |
| Create | `test/VertigoUnitTests/colgen/test_branching_strategy.jl` | Unit tests |
| Modify | `test/VertigoUnitTests/VertigoUnitTests.jl` | Include + register new test |

---

### Task 1: Create `BranchingCandidate` and `find_fractional_variables`

**Files:**
- Create: `src/BranchCutPrice/branching_candidates.jl`
- Create: `test/VertigoUnitTests/colgen/test_branching_strategy.jl`
- Modify: `test/VertigoUnitTests/VertigoUnitTests.jl`
- Modify: `src/BranchCutPrice/BranchCutPrice.jl`

- [ ] **Step 1: Write the test file with tests for `find_fractional_variables`**

Create `test/VertigoUnitTests/colgen/test_branching_strategy.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.BranchCutPrice: find_fractional_variables,
    BranchingCandidate

function test_branching_strategy()
    @testset "[find_fractional_variables] all integral" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)
        primal = Dict{MOI.VariableIndex,Float64}()
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        @test isempty(candidates)
    end

    @testset "[find_fractional_variables] detects fractional and sorts" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)
        primal = get_primal_solution(bp_master_model(ctx))
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        # GAP(2,5) always has fractional LP relaxation
        @test !isempty(candidates)
        c = first(candidates)
        @test c isa BranchingCandidate
        @test c.fractionality > 0.0
        @test c.floor_val == floor(c.value)
        @test c.ceil_val == ceil(c.value)
        # Verify sorted descending by fractionality
        for i in 1:length(candidates)-1
            @test candidates[i].fractionality >=
                  candidates[i+1].fractionality
        end
    end
end
```

Register in `test/VertigoUnitTests/VertigoUnitTests.jl`:
- Add import: `using Vertigo.BranchCutPrice: find_fractional_variables, BranchingCandidate, MostFractionalRule, LeastFractionalRule, select_candidates, MostFractionalBranching, select_branching_variable, bp_master_model` at line 19 (after the existing `using` statements).
- Add `include("colgen/test_branching_strategy.jl")` after the `test_max_cg_iterations.jl` include.
- Add `test_branching_strategy()` in `run()` after `test_max_cg_iterations()`.

- [ ] **Step 2: Create `branching_candidates.jl`**

Create `src/BranchCutPrice/branching_candidates.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    BranchingCandidate{X}

A fractional original variable eligible for branching.
"""
struct BranchingCandidate{X}
    orig_var::X
    value::Float64
    floor_val::Float64
    ceil_val::Float64
    fractionality::Float64
end

"""
    find_fractional_variables(ctx, primal_values; tol=1e-6)

Project master LP solution to original-variable space and return
all fractional variables as `BranchingCandidate`s, sorted by
fractionality descending (most fractional first).
"""
function find_fractional_variables(
    ctx,
    primal_values::Dict{MOI.VariableIndex,Float64};
    tol::Float64 = 1e-6
)
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)
    x_values = project_to_original(
        decomp, pool, v -> get(primal_values, v, 0.0)
    )

    candidates = BranchingCandidate[]
    for (orig_var, x_val) in x_values
        frac_part = x_val - floor(x_val)
        (frac_part < tol || frac_part > 1.0 - tol) && continue
        fractionality = min(frac_part, 1.0 - frac_part)
        push!(candidates, BranchingCandidate(
            orig_var, x_val, floor(x_val), ceil(x_val),
            fractionality
        ))
    end
    sort!(candidates; by=c -> c.fractionality, rev=true)
    return candidates
end
```

- [ ] **Step 3: Add include in `BranchCutPrice.jl`**

In `src/BranchCutPrice/BranchCutPrice.jl`, add after `include("branching.jl")` (line 18):

```julia
include("branching_candidates.jl")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.test_branching_strategy()"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/BranchCutPrice/branching_candidates.jl src/BranchCutPrice/BranchCutPrice.jl test/VertigoUnitTests/colgen/test_branching_strategy.jl test/VertigoUnitTests/VertigoUnitTests.jl
git commit -m "add: BranchingCandidate and find_fractional_variables"
```

---

### Task 2: Create branching rules

**Files:**
- Create: `src/BranchCutPrice/branching_rules.jl`
- Modify: `src/BranchCutPrice/BranchCutPrice.jl`
- Modify: `test/VertigoUnitTests/colgen/test_branching_strategy.jl`

- [ ] **Step 1: Add tests for branching rules**

Append to `test_branching_strategy()` in `test_branching_strategy.jl`:

```julia
    @testset "[branching_rules] MostFractionalRule ordering" begin
        # Manually create candidates with known fractionalities
        c1 = BranchingCandidate(1, 1.3, 1.0, 2.0, 0.3)
        c2 = BranchingCandidate(2, 2.5, 2.0, 3.0, 0.5)
        c3 = BranchingCandidate(3, 3.1, 3.0, 4.0, 0.1)
        candidates = [c2, c1, c3]  # already sorted desc
        result = select_candidates(
            MostFractionalRule(), candidates, 2
        )
        @test length(result) == 2
        @test result[1].fractionality == 0.5
        @test result[2].fractionality == 0.3
    end

    @testset "[branching_rules] LeastFractionalRule ordering" begin
        c1 = BranchingCandidate(1, 1.3, 1.0, 2.0, 0.3)
        c2 = BranchingCandidate(2, 2.5, 2.0, 3.0, 0.5)
        c3 = BranchingCandidate(3, 3.1, 3.0, 4.0, 0.1)
        candidates = [c2, c1, c3]
        result = select_candidates(
            LeastFractionalRule(), candidates, 2
        )
        @test length(result) == 2
        @test result[1].fractionality == 0.1
        @test result[2].fractionality == 0.3
    end

    @testset "[branching_rules] max_candidates truncation" begin
        cs = [BranchingCandidate(i, Float64(i) + 0.5, Float64(i), Float64(i) + 1.0, 0.5) for i in 1:10]
        result = select_candidates(MostFractionalRule(), cs, 3)
        @test length(result) == 3
    end
```

- [ ] **Step 2: Create `branching_rules.jl`**

Create `src/BranchCutPrice/branching_rules.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    AbstractBranchingRule

Determines candidate ordering for branching variable selection.
"""
abstract type AbstractBranchingRule end

"""
    MostFractionalRule <: AbstractBranchingRule

Select candidates closest to 0.5 fractionality first.
"""
struct MostFractionalRule <: AbstractBranchingRule end

"""
    LeastFractionalRule <: AbstractBranchingRule

Select candidates closest to integrality first.
"""
struct LeastFractionalRule <: AbstractBranchingRule end

"""
    select_candidates(rule, candidates, max_candidates)

Return the top `max_candidates` from `candidates` ordered by `rule`.
Input `candidates` is assumed sorted by fractionality descending.
"""
function select_candidates(
    ::MostFractionalRule,
    candidates::Vector{<:BranchingCandidate},
    max_candidates::Int
)
    return candidates[1:min(length(candidates), max_candidates)]
end

function select_candidates(
    ::LeastFractionalRule,
    candidates::Vector{<:BranchingCandidate},
    max_candidates::Int
)
    sorted = sort(candidates; by=c -> c.fractionality)
    return sorted[1:min(length(sorted), max_candidates)]
end
```

- [ ] **Step 3: Add include in `BranchCutPrice.jl`**

After `include("branching_candidates.jl")`:

```julia
include("branching_rules.jl")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.test_branching_strategy()"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/BranchCutPrice/branching_rules.jl src/BranchCutPrice/BranchCutPrice.jl test/VertigoUnitTests/colgen/test_branching_strategy.jl
git commit -m "add: AbstractBranchingRule with MostFractional and LeastFractional"
```

---

### Task 3: Create strategy interface and `MostFractionalBranching`

**Files:**
- Create: `src/BranchCutPrice/branching_strategy.jl`
- Modify: `src/BranchCutPrice/BranchCutPrice.jl`
- Modify: `test/VertigoUnitTests/colgen/test_branching_strategy.jl`

- [ ] **Step 1: Add test for `MostFractionalBranching`**

Append to `test_branching_strategy()`:

```julia
    @testset "[branching_strategy] MostFractionalBranching delegates" begin
        inst = random_gap_instance(2, 5)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)
        primal = get_primal_solution(bp_master_model(ctx))

        # Compare strategy with direct call (same tol)
        orig_var, x_val = most_fractional_original_variable(
            ctx, primal; tol=1e-6
        )
        # Strategy needs a BPSpace — construct one
        space = BPSpace(ctx; node_limit=1)
        result = select_branching_variable(
            MostFractionalBranching(), space, nothing, primal
        )
        # GAP(2,5) always has fractional LP relaxation
        @test !isnothing(orig_var)
        @test !isnothing(result)
        @test result[1] == orig_var
        @test result[2] ≈ x_val
    end
```

Add `most_fractional_original_variable` to the imports in `VertigoUnitTests.jl`:

```julia
using Vertigo.BranchCutPrice: find_fractional_variables,
    BranchingCandidate, MostFractionalRule, LeastFractionalRule,
    select_candidates, MostFractionalBranching,
    select_branching_variable, bp_master_model,
    most_fractional_original_variable, BPSpace
```

- [ ] **Step 2: Create `branching_strategy.jl`**

Create `src/BranchCutPrice/branching_strategy.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    AbstractBranchingStrategy

Determines how to select a branching variable at each node.
"""
abstract type AbstractBranchingStrategy end

"""
    MostFractionalBranching <: AbstractBranchingStrategy

Select the most fractional original variable for branching.
Delegates to `most_fractional_original_variable`.
"""
struct MostFractionalBranching <: AbstractBranchingStrategy end

"""
    select_branching_variable(strategy, space, node, primal_values)

Select a variable to branch on. Returns `(orig_var, x_val)` or
`nothing` if all variables are integral. Tolerance comes from
`space.tol`.
"""
function select_branching_variable(
    ::MostFractionalBranching, space, node,
    primal_values::Dict{MOI.VariableIndex,Float64}
)
    orig_var, x_val = most_fractional_original_variable(
        space.ctx, primal_values; tol=space.tol
    )
    isnothing(orig_var) && return nothing
    return (orig_var, x_val)
end
```

- [ ] **Step 3: Add include and exports in `BranchCutPrice.jl`**

After `include("branching_rules.jl")`:

```julia
include("branching_strategy.jl")
```

Add exports after the existing `export` lines:

```julia
export AbstractBranchingStrategy, MostFractionalBranching
export AbstractBranchingRule, MostFractionalRule, LeastFractionalRule
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.test_branching_strategy()"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/BranchCutPrice/branching_strategy.jl src/BranchCutPrice/BranchCutPrice.jl test/VertigoUnitTests/colgen/test_branching_strategy.jl test/VertigoUnitTests/VertigoUnitTests.jl
git commit -m "add: AbstractBranchingStrategy with MostFractionalBranching"
```

---

### Task 4: Refactor `BPSpace` and `branch!` to use strategy

**Files:**
- Modify: `src/BranchCutPrice/space.jl`
- Modify: `test/VertigoUnitTests/colgen/test_branching_strategy.jl`

- [ ] **Step 1: Add `branching_strategy` field to `BPSpace`**

In `src/BranchCutPrice/space.jl`, add `branching_strategy::AbstractBranchingStrategy` field after `total_cuts_separated::Int` (line 46):

```julia
    total_cuts_separated::Int
    branching_strategy::AbstractBranchingStrategy
```

Update `BPSpace` constructor signature (line 57-66) to accept the kwarg:

```julia
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
```

Add `branching_strategy` at the end of the `BPSpace(...)` call inside the constructor (line 76-88):

```julia
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
        0, branching_strategy
    )
```

- [ ] **Step 2: Refactor `branch!` to use `select_branching_variable`**

In `src/BranchCutPrice/space.jl`, replace lines 152-158 in `TreeSearch.branch!`:

From:
```julia
function TreeSearch.branch!(space::BPSpace, node)
    primal_values = get_primal_solution(space.backend)

    orig_var, x_val = most_fractional_original_variable(
        space.ctx, primal_values; tol = space.tol
    )
    isnothing(orig_var) && return typeof(node)[]
```

To:
```julia
function TreeSearch.branch!(space::BPSpace, node)
    primal_values = get_primal_solution(space.backend)

    result = select_branching_variable(
        space.branching_strategy, space, node,
        primal_values
    )
    isnothing(result) && return typeof(node)[]
    orig_var, x_val = result
```

- [ ] **Step 3: Add `branching_strategy` kwarg to `run_branch_and_price`**

In `src/BranchCutPrice/space.jl`, add kwarg to `run_branch_and_price` signature (after `min_gap_improvement`):

```julia
    branching_strategy::AbstractBranchingStrategy = MostFractionalBranching(),
```

And forward it in the `BPSpace(...)` call inside `run_branch_and_price`:

```julia
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
```

- [ ] **Step 4: Run all unit tests for regression**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.run()"`
Expected: All PASS (default `MostFractionalBranching` gives same behavior)

- [ ] **Step 5: Run full regression (unit + e2e)**

Run: `julia --project=. -e "include(\"test/runtests.jl\")"`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/BranchCutPrice/space.jl
git commit -m "refactor: use pluggable branching strategy in BPSpace"
```
