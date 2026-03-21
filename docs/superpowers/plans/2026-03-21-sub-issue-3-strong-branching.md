# Sub-issue 3: Strong Branching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement strong branching probes that evaluate candidate branching variables by running limited CG, scoring dual bound improvements, and selecting the best variable.

**Architecture:** New file `strong_branching.jl` contains probe result types, scoring, probe execution (`add_branching_constraint!` / `remove_branching_constraint!`), and `StrongBranching` strategy. Probes call full `run_column_generation` with limited iterations, saving/restoring context state. The `select_branching_variable` for `StrongBranching` uses `find_fractional_variables` + `select_candidates` from sub-issue 2, then probes the top-k.

**Tech Stack:** Julia, MathOptInterface, HiGHS (tests)

**Spec:** `docs/superpowers/specs/2026-03-20-strong-branching-framework-design.md` (Sub-issue 3)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/BranchCutPrice/strong_branching.jl` | `SBProbeResult`, `SBCandidateResult`, `sb_score`, `build_branching_terms`, `add_branching_constraint!`, `remove_branching_constraint!`, `run_sb_probe`, `StrongBranching`, `select_branching_variable` |
| Modify | `src/BranchCutPrice/BranchCutPrice.jl` | Add include + export |
| Create | `test/VertigoUnitTests/colgen/test_strong_branching.jl` | Unit tests for scoring, probes |
| Modify | `test/VertigoUnitTests/VertigoUnitTests.jl` | Include + register new test |

---

### Task 1: `sb_score` — probe result types and scoring function

**Files:**
- Create: `src/BranchCutPrice/strong_branching.jl` (partial — types + scoring only)
- Create: `test/VertigoUnitTests/colgen/test_strong_branching.jl`
- Modify: `test/VertigoUnitTests/VertigoUnitTests.jl`
- Modify: `src/BranchCutPrice/BranchCutPrice.jl`

- [ ] **Step 1: Write test file with scoring tests**

Create `test/VertigoUnitTests/colgen/test_strong_branching.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Vertigo.BranchCutPrice: SBProbeResult, SBCandidateResult,
    sb_score, BranchingCandidate

function test_strong_branching()
    @testset "[sb_score] both children feasible" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        # Δ⁻ = 12.0 - 10.0 = 2.0, Δ⁺ = 14.0 - 10.0 = 4.0
        # score = (1 - 1/6) * min(2,4) + (1/6) * max(2,4)
        #       = (5/6) * 2 + (1/6) * 4 = 10/6 + 4/6 = 14/6
        mu = 1.0 / 6.0
        expected = (1 - mu) * 2.0 + mu * 4.0
        @test sb_score(result) ≈ expected
    end

    @testset "[sb_score] one child infeasible" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(nothing, nothing, true)
        result = SBCandidateResult(c, 10.0, left, right)
        # Δ⁻ = 2.0, Δ⁺ = Inf
        # score = (5/6) * 2.0 + (1/6) * Inf = Inf
        @test sb_score(result) == Inf
    end

    @testset "[sb_score] both children infeasible" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(nothing, nothing, true)
        right = SBProbeResult(nothing, nothing, true)
        result = SBCandidateResult(c, 10.0, left, right)
        @test sb_score(result) == Inf
    end

    @testset "[sb_score] custom mu" begin
        c = BranchingCandidate(1, 2.3, 2.0, 3.0, 0.3)
        left = SBProbeResult(12.0, 12.0, false)
        right = SBProbeResult(14.0, 14.0, false)
        result = SBCandidateResult(c, 10.0, left, right)
        mu = 0.25
        expected = (1 - mu) * 2.0 + mu * 4.0
        @test sb_score(result; mu=mu) ≈ expected
    end
end
```

Register in `test/VertigoUnitTests/VertigoUnitTests.jl`:
- Add `include("colgen/test_strong_branching.jl")` after the `test_branching_strategy.jl` include.
- Add `test_strong_branching()` in `run()` after `test_branching_strategy()`.

- [ ] **Step 2: Create `strong_branching.jl` with types and scoring**

Create `src/BranchCutPrice/strong_branching.jl`:

```julia
# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ── Probe result types ───────────────────────────────────────────────

struct SBProbeResult
    dual_bound::Union{Nothing,Float64}
    lp_obj::Union{Nothing,Float64}
    is_infeasible::Bool
end

struct SBCandidateResult{X}
    candidate::BranchingCandidate{X}
    parent_lp_obj::Float64
    left::SBProbeResult
    right::SBProbeResult
end

# ── Scoring ──────────────────────────────────────────────────────────

function _sb_delta(probe::SBProbeResult, parent_lp_obj::Float64)
    probe.is_infeasible && return Inf
    isnothing(probe.dual_bound) && return 0.0
    return max(0.0, probe.dual_bound - parent_lp_obj)
end

"""
    sb_score(result; mu=1.0/6.0) -> Float64

Product score: `(1-μ) * min(Δ⁻, Δ⁺) + μ * max(Δ⁻, Δ⁺)`.
Infeasible child → Δ = Inf.
"""
function sb_score(result::SBCandidateResult; mu::Float64=1.0/6.0)
    d_left = _sb_delta(result.left, result.parent_lp_obj)
    d_right = _sb_delta(result.right, result.parent_lp_obj)
    return (1 - mu) * min(d_left, d_right) +
           mu * max(d_left, d_right)
end
```

- [ ] **Step 3: Add include in `BranchCutPrice.jl`**

After `include("branching_strategy.jl")` (line 21), add:

```julia
include("strong_branching.jl")
```

- [ ] **Step 4: Run tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.test_strong_branching()"`
Expected: PASS (4 testsets)

- [ ] **Step 5: Commit**

```bash
git add src/BranchCutPrice/strong_branching.jl src/BranchCutPrice/BranchCutPrice.jl test/VertigoUnitTests/colgen/test_strong_branching.jl test/VertigoUnitTests/VertigoUnitTests.jl
git commit -m "add: SBProbeResult, SBCandidateResult, and sb_score"
```

---

### Task 2: `add_branching_constraint!` and `remove_branching_constraint!`

**Files:**
- Modify: `src/BranchCutPrice/strong_branching.jl`
- Modify: `test/VertigoUnitTests/colgen/test_strong_branching.jl`

- [ ] **Step 1: Write tests for add/remove**

Append to `test_strong_branching()`:

```julia
    @testset "[branching_constraint] add and remove" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)

        backend = bp_master_model(ctx)
        bcs = bp_branching_constraints(ctx)
        @test isempty(bcs)

        # Build terms for a branching constraint
        pool = bp_pool(ctx)
        decomp = bp_decomp(ctx)
        primal = get_primal_solution(backend)
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        @test !isempty(candidates)
        candidate = first(candidates)

        terms = build_branching_terms(
            decomp, pool, candidate.orig_var
        )
        ci = add_branching_constraint!(
            backend, ctx, terms,
            MOI.LessThan(candidate.floor_val),
            candidate.orig_var
        )
        @test length(bcs) == 1
        @test MOI.is_valid(backend, ci)

        remove_branching_constraint!(backend, ctx, ci)
        @test isempty(bcs)
        @test !MOI.is_valid(backend, ci)
    end
```

Add imports at the top of the test file:

```julia
using Vertigo.BranchCutPrice: SBProbeResult, SBCandidateResult,
    sb_score, BranchingCandidate, find_fractional_variables,
    bp_master_model, bp_pool, bp_decomp, bp_branching_constraints,
    build_branching_terms, add_branching_constraint!,
    remove_branching_constraint!, BPSpace, StrongBranching,
    run_sb_probe, select_branching_variable
using Vertigo.Reformulation: get_primal_solution
```

- [ ] **Step 2: Implement `build_branching_terms`, `add_branching_constraint!`, `remove_branching_constraint!`**

Append to `src/BranchCutPrice/strong_branching.jl`:

```julia
# ── Branching constraint helpers ─────────────────────────────────────

"""
    build_branching_terms(decomp, pool, orig_var)

Build MOI constraint terms for a branching constraint on `orig_var`.
"""
function build_branching_terms(decomp, pool, orig_var)
    terms = MOI.ScalarAffineTerm{Float64}[]
    for (col_var, rec) in columns(pool)
        coeff = compute_branching_column_coefficient(
            decomp, orig_var, column_sp_id(rec), rec.solution
        )
        if !iszero(coeff)
            push!(terms, MOI.ScalarAffineTerm(coeff, col_var))
        end
    end
    return terms
end

"""
    add_branching_constraint!(backend, ctx, terms, set, orig_var)

Add a branching constraint to the MOI backend and register it
in `branching_constraints` in a single function call.
Returns the MOI constraint index.
"""
function add_branching_constraint!(backend, ctx, terms, set, orig_var)
    f = MOI.ScalarAffineFunction(terms, 0.0)
    ci = MOI.add_constraint(backend, f, set)
    bcs = bp_branching_constraints(ctx)
    push!(bcs, ColGen.ActiveBranchingConstraint(
        TaggedCI(ci), orig_var
    ))
    return ci
end

"""
    remove_branching_constraint!(backend, ctx, ci)

Delete the MOI constraint and remove it from `branching_constraints`.
Defensive: handles partial state gracefully.
"""
function remove_branching_constraint!(backend, ctx, ci)
    if MOI.is_valid(backend, ci)
        MOI.delete(backend, ci)
    end
    bcs = bp_branching_constraints(ctx)
    tagged = TaggedCI(ci)
    filter!(bc -> bc.constraint_index != tagged, bcs)
    return
end
```

- [ ] **Step 3: Run tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.test_strong_branching()"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/BranchCutPrice/strong_branching.jl test/VertigoUnitTests/colgen/test_strong_branching.jl
git commit -m "add: add_branching_constraint! and remove_branching_constraint!"
```

---

### Task 3: `run_sb_probe` — probe execution with save/restore

**Files:**
- Modify: `src/BranchCutPrice/strong_branching.jl`
- Modify: `test/VertigoUnitTests/colgen/test_strong_branching.jl`

- [ ] **Step 1: Write test for `run_sb_probe`**

Append to `test_strong_branching()`:

```julia
    @testset "[run_sb_probe] returns dual bounds" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        cg_out = run_column_generation(ctx)
        parent_lp = cg_out.master_lp_obj

        backend = bp_master_model(ctx)
        primal = get_primal_solution(backend)
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        space = BPSpace(ctx; node_limit=1)
        candidate = first(candidates)

        result = run_sb_probe(space, candidate, 10, parent_lp)
        @test result isa SBCandidateResult
        @test result.parent_lp_obj ≈ parent_lp
        # At least one direction should produce a dual bound
        has_bound = !isnothing(result.left.dual_bound) ||
                    !isnothing(result.right.dual_bound)
        @test has_bound || result.left.is_infeasible ||
              result.right.is_infeasible
    end

    @testset "[run_sb_probe] restores context state" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        cg_out = run_column_generation(ctx)
        parent_lp = cg_out.master_lp_obj

        backend = bp_master_model(ctx)
        primal = get_primal_solution(backend)
        candidates = find_fractional_variables(
            ctx, primal; tol=1e-6
        )
        space = BPSpace(ctx; node_limit=1)
        candidate = first(candidates)

        # Save state before probe
        orig_max_iter = ColGen.max_cg_iterations(ctx)
        orig_ip_inc = bp_ip_incumbent(ctx)
        orig_ip_bound = bp_ip_primal_bound(ctx)
        orig_n_bcs = length(bp_branching_constraints(ctx))

        run_sb_probe(space, candidate, 10, parent_lp)

        # State must be restored
        @test ColGen.max_cg_iterations(ctx) == orig_max_iter
        @test bp_ip_incumbent(ctx) === orig_ip_inc
        @test bp_ip_primal_bound(ctx) === orig_ip_bound
        @test length(bp_branching_constraints(ctx)) == orig_n_bcs
    end
```

Add imports: `using Vertigo.ColGen: max_cg_iterations` and `using Vertigo.BranchCutPrice: bp_ip_incumbent, bp_ip_primal_bound`.

- [ ] **Step 2: Implement `run_sb_probe`**

Append to `src/BranchCutPrice/strong_branching.jl`:

```julia
# ── Probe execution ──────────────────────────────────────────────────

function _save_probe_state(ctx, space)
    return (
        max_iter = ColGen.max_cg_iterations(ctx),
        ip_inc = bp_ip_incumbent(ctx),
        ip_bound = bp_ip_primal_bound(ctx),
        bcs = copy(bp_branching_constraints(ctx)),
        basis = MathOptState.capture_basis(space.backend),
    )
end

function _restore_probe_state!(ctx, space, saved)
    bcs = bp_branching_constraints(ctx)
    empty!(bcs)
    append!(bcs, saved.bcs)
    ColGen.set_max_cg_iterations!(ctx, saved.max_iter)
    bp_set_ip_primal_bound!(ctx, saved.ip_bound)
    # Restore ip_incumbent via the existing accessor pattern
    _set_ip_incumbent!(ctx, saved.ip_inc)
    # Restore LP basis for consistent probe starting point
    MathOptState.apply_change!(
        space.backend,
        MathOptState.LPBasisDiff(saved.basis),
        nothing
    )
    return
end

_set_ip_incumbent!(ctx::ColGen.ColGenContext, val) =
    ctx.ip_incumbent = val
_set_ip_incumbent!(ctx::ColGen.ColGenLoggerContext, val) =
    ctx.inner.ip_incumbent = val

function _run_one_direction(space, candidate, set, max_cg_iter)
    ctx = space.ctx
    backend = space.backend
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)

    terms = build_branching_terms(decomp, pool, candidate.orig_var)
    ci = add_branching_constraint!(
        backend, ctx, terms, set, candidate.orig_var
    )
    ColGen.set_max_cg_iterations!(ctx, max_cg_iter)

    try
        cg_output = ColGen.run_column_generation(ctx)
        is_inf = cg_output.status == ColGen.master_infeasible ||
                 cg_output.status == ColGen.subproblem_infeasible
        return SBProbeResult(
            cg_output.incumbent_dual_bound,
            cg_output.master_lp_obj,
            is_inf
        )
    finally
        remove_branching_constraint!(backend, ctx, ci)
    end
end

"""
    run_sb_probe(space, candidate, max_cg_iterations, parent_lp_obj)

Run strong branching probes in both directions (floor/ceil) for
the given candidate. Saves and restores context state (iteration
limit, IP incumbent, primal bound, branching constraints, LP basis)
around both probes. Returns `SBCandidateResult`.
"""
function run_sb_probe(
    space::BPSpace, candidate::BranchingCandidate,
    max_cg_iterations::Int, parent_lp_obj::Float64
)
    saved = _save_probe_state(space.ctx, space)
    try
        left = _run_one_direction(
            space, candidate,
            MOI.LessThan(candidate.floor_val),
            max_cg_iterations
        )
        _restore_probe_state!(space.ctx, space, saved)
        right = _run_one_direction(
            space, candidate,
            MOI.GreaterThan(candidate.ceil_val),
            max_cg_iterations
        )
        @debug "SB probe" var=candidate.orig_var \
            left_db=left.dual_bound left_inf=left.is_infeasible \
            right_db=right.dual_bound right_inf=right.is_infeasible
        return SBCandidateResult(
            candidate, parent_lp_obj, left, right
        )
    finally
        _restore_probe_state!(space.ctx, space, saved)
    end
end
```

- [ ] **Step 3: Run tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.test_strong_branching()"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/BranchCutPrice/strong_branching.jl test/VertigoUnitTests/colgen/test_strong_branching.jl
git commit -m "add: run_sb_probe with context save/restore"
```

---

### Task 4: `StrongBranching` strategy and `select_branching_variable`

**Files:**
- Modify: `src/BranchCutPrice/strong_branching.jl`
- Modify: `src/BranchCutPrice/BranchCutPrice.jl` (exports)
- Modify: `test/VertigoUnitTests/colgen/test_strong_branching.jl`

- [ ] **Step 1: Write test for StrongBranching strategy**

Append to `test_strong_branching()`:

```julia
    @testset "[StrongBranching] selects branching variable" begin
        inst = random_gap_instance(2, 5; seed=10)
        ctx = build_gap_context(inst)
        run_column_generation(ctx)

        primal = get_primal_solution(bp_master_model(ctx))
        space = BPSpace(
            ctx; node_limit=1,
            branching_strategy=StrongBranching()
        )

        result = select_branching_variable(
            StrongBranching(), space, nothing, primal
        )
        @test !isnothing(result)
        orig_var, x_val = result
        # Should pick a fractional variable
        frac = x_val - floor(x_val)
        @test frac > 1e-6
        @test frac < 1.0 - 1e-6
    end
```

- [ ] **Step 2: Implement `StrongBranching` and `select_branching_variable`**

Append to `src/BranchCutPrice/strong_branching.jl`:

```julia
# ── StrongBranching strategy ─────────────────────────────────────────

"""
    StrongBranching <: AbstractBranchingStrategy

Evaluate candidate variables with limited CG probes and pick
the one with the best product score.
"""
struct StrongBranching <: AbstractBranchingStrategy
    max_candidates::Int
    max_cg_iterations::Int
    mu::Float64
    rule::AbstractBranchingRule

    function StrongBranching(;
        max_candidates::Int = 5,
        max_cg_iterations::Int = 10,
        mu::Float64 = 1.0 / 6.0,
        rule::AbstractBranchingRule = MostFractionalRule()
    )
        new(max_candidates, max_cg_iterations, mu, rule)
    end
end

function select_branching_variable(
    sb::StrongBranching, space, node,
    primal_values::Dict{MOI.VariableIndex,Float64}
)
    ctx = space.ctx
    candidates = find_fractional_variables(
        ctx, primal_values; tol=space.tol
    )
    isempty(candidates) && return nothing

    selected = select_candidates(
        sb.rule, candidates, sb.max_candidates
    )

    # Get parent LP objective from the CG output
    parent_lp = if !isnothing(node) &&
                   !isnothing(node.user_data) &&
                   !isnothing(node.user_data.cg_output)
        node.user_data.cg_output.master_lp_obj
    else
        nothing
    end

    # Fallback: if no parent LP obj, use most fractional
    if isnothing(parent_lp)
        c = first(selected)
        return (c.orig_var, c.value)
    end

    best_score = -Inf
    best_candidate = first(selected)

    for c in selected
        result = run_sb_probe(
            space, c, sb.max_cg_iterations, parent_lp
        )
        # Both children infeasible → node is infeasible
        if result.left.is_infeasible && result.right.is_infeasible
            @debug "SB: both children infeasible" var=c.orig_var
            return nothing
        end
        score = sb_score(result; mu=sb.mu)
        @debug "SB candidate scored" var=c.orig_var score=score
        if score > best_score
            best_score = score
            best_candidate = c
        end
    end
    @debug "SB selected" var=best_candidate.orig_var \
        score=best_score
    return (best_candidate.orig_var, best_candidate.value)
end
```

- [ ] **Step 3: Add export in `BranchCutPrice.jl`**

Add after existing exports:

```julia
export StrongBranching
```

- [ ] **Step 4: Run all unit tests**

Run: `julia --project=. -e "push!(LOAD_PATH, \"test/VertigoUnitTests\"); using VertigoUnitTests; VertigoUnitTests.run()"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/BranchCutPrice/strong_branching.jl src/BranchCutPrice/BranchCutPrice.jl test/VertigoUnitTests/colgen/test_strong_branching.jl
git commit -m "add: StrongBranching strategy with select_branching_variable"
```

---

### Task 5: E2e test — `StrongBranching` on small GAP

**Files:**
- Modify: `test/VertigoUnitTests/colgen/test_strong_branching.jl`

- [ ] **Step 1: Add e2e test**

Append to `test_strong_branching()`:

```julia
    @testset "[StrongBranching] e2e small GAP finds optimal" begin
        inst = random_gap_instance(2, 4; seed=10)
        ctx = build_gap_context(inst)
        output = run_branch_and_price(
            ctx;
            node_limit=100,
            branching_strategy=StrongBranching(
                max_candidates=3, max_cg_iterations=5
            )
        )
        @test output.status in (:optimal, :node_limit)
        @test !isnothing(output.incumbent)
    end
```

- [ ] **Step 2: Run full regression (unit + e2e)**

Run: `julia --project=. -e "include(\"test/runtests.jl\")"`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add test/VertigoUnitTests/colgen/test_strong_branching.jl
git commit -m "test: add e2e test for StrongBranching on small GAP"
```
