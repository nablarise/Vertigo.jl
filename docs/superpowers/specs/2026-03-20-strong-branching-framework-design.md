# Strong Branching Framework for Branch-and-Price

**Date:** 2026-03-20
**Issue:** #17
**Status:** Design approved

---

## Overview

Vertigo currently uses a single hard-coded branching rule (`most_fractional_original_variable`) that picks the variable closest to 0.5. This design introduces a pluggable branching strategy interface, strong branching with CG probes, and reliability branching with pseudocost tracking.

The work is decomposed into 4 independent sub-issues, each merged separately.

---

## Sub-issue 1: Configurable `max_cg_iterations` on `ColGenContext`

**Prerequisite for:** Sub-issue 3 (strong branching probes need limited CG runs).

### Changes

- Add `max_cg_iterations::Int` field to `ColGenContext` (default `1000`).
- Add `max_cg_iterations(ctx)` accessor and `set_max_cg_iterations!(ctx, n)` mutator.
- Replace the hardcoded `1000` in `stop_colgen_phase` (`context.jl:386`) with `max_cg_iterations(ctx)`.
- Expose as a kwarg on `ColGenContext` constructor.
- Wire through `ColGenLoggerContext` with the same accessor/mutator pattern.

### Scope boundaries

- Phase 1 (feasibility) keeps no iteration limit — only Phase 0 and Phase 2 are affected.
- The limit applies **per phase**, not globally. A `max_cg_iterations=10` allows up to 10 iterations in Phase 0 and 10 in Phase 2 (20 total, Phase 1 unlimited). This is intentional: it matches the existing hardcoded behavior (1000 per phase).
- All existing behavior is preserved (default is 1000).

### Semantic note

This field serves double duty: it configures the normal CG iteration limit and is also used by strong branching probes (sub-issue 3) which save/restore it temporarily. The save/restore pattern ensures probes don't leak their low budget to normal CG runs. Users configuring `ColGenContext` directly should set the limit for normal CG behavior; probes manage their own budget internally.

### Tests

- Unit: construct `ColGenContext` with custom limit, verify `stop_colgen_phase` respects it.
- Existing e2e tests pass unchanged (default 1000).

### Files

| Action | File |
|--------|------|
| Modify | `src/ColGen/context.jl` — struct field, constructor, accessor, mutator, stop logic |
| Modify | `src/ColGen/logger.jl` — forward `set_max_cg_iterations!` to inner context |

---

## Sub-issue 2: Branching candidates, rules, and pluggable strategy interface

**Depends on:** nothing. **Prerequisite for:** Sub-issue 3.

### New types

#### `BranchingCandidate{X}`

```julia
struct BranchingCandidate{X}
    orig_var::X
    value::Float64
    floor_val::Float64
    ceil_val::Float64
    fractionality::Float64  # min(frac, 1 - frac)
end
```

#### Branching rules

```julia
abstract type AbstractBranchingRule end
struct MostFractionalRule <: AbstractBranchingRule end
struct LeastFractionalRule <: AbstractBranchingRule end

function select_candidates(
    rule::AbstractBranchingRule,
    candidates::Vector{BranchingCandidate},
    max_candidates::Int
) -> Vector{BranchingCandidate}
```

`MostFractionalRule` returns the first `max_candidates` (already sorted most-fractional-first). `LeastFractionalRule` reverses the order. Pure functions, no side effects.

#### Strategy interface

```julia
abstract type AbstractBranchingStrategy end

struct MostFractionalBranching <: AbstractBranchingStrategy end

function select_branching_variable(
    ::MostFractionalBranching, space, node, primal_values
) -> Union{Nothing, Tuple{Any, Float64}}
```

`MostFractionalBranching` delegates to the existing `most_fractional_original_variable` — zero behavior change.

### Functions

- `find_fractional_variables(ctx, primal_values; tol=1e-6)` — reuses `project_to_original`, returns `Vector{BranchingCandidate}` sorted by fractionality descending.

### Refactor `BPSpace` and `branch!`

- Add `branching_strategy::AbstractBranchingStrategy` field to `BPSpace` (default `MostFractionalBranching()`).
- Add `branching_strategy` kwarg to `BPSpace` constructor and `run_branch_and_price`.
- `TreeSearch.branch!` replaces its direct call to `most_fractional_original_variable` with `select_branching_variable(space.branching_strategy, space, node, primal_values)`.

### Include order in `BranchCutPrice.jl`

```
interface.jl → bp_output.jl → branching.jl →
branching_candidates.jl → branching_rules.jl → branching_strategy.jl →
cut_col_gen.jl → space.jl → ...
```

### Exports

```julia
export AbstractBranchingStrategy, MostFractionalBranching
export AbstractBranchingRule, MostFractionalRule, LeastFractionalRule
```

### Tests

- Unit: `find_fractional_variables` — all integral, single fractional, multiple.
- Unit: `MostFractionalRule` / `LeastFractionalRule` — ordering, `max_candidates` truncation.
- Unit: `MostFractionalBranching` delegates correctly.
- Existing e2e tests pass unchanged (default strategy).

### Files

| Action | File |
|--------|------|
| Create | `src/BranchCutPrice/branching_candidates.jl` |
| Create | `src/BranchCutPrice/branching_rules.jl` |
| Create | `src/BranchCutPrice/branching_strategy.jl` |
| Modify | `src/BranchCutPrice/BranchCutPrice.jl` — includes + exports |
| Modify | `src/BranchCutPrice/space.jl` — new field, kwarg, refactored `branch!` |

---

## Sub-issue 3: Strong branching probes and scoring

**Depends on:** Sub-issues 1 and 2.

### Probe result types

```julia
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
```

### Scoring

```julia
function sb_score(result::SBCandidateResult; mu=1.0/6.0) -> Float64
```

Standard product score: `(1-μ) * min(Δ⁻, Δ⁺) + μ * max(Δ⁻, Δ⁺)` where `Δ` is the dual bound improvement over the parent LP objective. Infeasible child → `Δ = Inf`.

### Probe execution

```julia
function run_sb_probe(
    space::BPSpace, candidate::BranchingCandidate,
    max_cg_iterations::Int
) -> SBCandidateResult
```

For each direction (floor/ceil):
1. Save current `max_cg_iterations(ctx)`, `ctx.ip_incumbent`, and `ctx.ip_primal_bound`.
2. `set_max_cg_iterations!(ctx, probe_budget)`.
3. `add_branching_constraint!(backend, ctx, terms, set, orig_var)` — adds the MOI constraint and registers it in `branching_constraints` in a single function call.
4. `ColGen.run_column_generation(ctx)` — full CG with Phase 0 → 1 → 2. CG creates and manages its own artificial variables as usual. The branching constraint may make the restricted master infeasible; Phase 0/1 handle feasibility recovery, Phase 2 optimizes. The iteration limit applies per phase.
5. Record dual bound, LP obj, infeasibility from CG output.
6. In `finally`: `remove_branching_constraint!(backend, ctx, ci)` — deletes the MOI constraint and removes it from `branching_constraints`. Restore `max_cg_iterations`, `ip_incumbent`, and `ip_primal_bound`.

**Both children infeasible:** If the left probe returns infeasible and the right probe also returns infeasible, the parent node is itself infeasible. `run_sb_probe` detects this and returns early with a result indicating node infeasibility, without evaluating remaining candidates.

**Why full `run_column_generation`:** The existing CG already handles artificial variable creation (Phase 0), feasibility recovery (Phase 1), and optimization (Phase 2). Probes reuse this machinery as-is rather than reimplementing phase logic. The overhead of Phase 0/1 on all constraints is acceptable for probes.

**Why not use `LocalCutTracker`:** Probe constraints are ephemeral. Routing them through `apply_change!` would add the probe cut ID to `cut_helper.active_cuts`, and `_rebuild_branching_constraints!` would then crash with a `KeyError` because the probe cut ID is not in `branching_cut_info`. Using `MOI.add_constraint`/`MOI.delete` directly avoids this.

**Exception safety:** `remove_branching_constraint!` must be defensive: if the MOI constraint exists, delete it; if the entry is in `branching_constraints`, remove it. This handles partial state if `add_branching_constraint!` succeeded on the MOI side but failed before registering (unlikely but possible).

**Context state save/restore:** Probes can discover IP-feasible solutions or update the primal bound. Since these side effects could mislead the parent node's subsequent branching decision, we save and restore `ip_incumbent` and `ip_primal_bound`. Columns discovered during probes remain in the pool (beneficial).

**Column warm-start:** Each probe starts with all columns from the parent's pool already in the master restricted problem. The pool is global and persists across probes — pricing only adds new columns. This means probes start from a good restricted master and 10 CG iterations is typically sufficient to get a meaningful dual bound.

**LP basis between probes:** After each probe, the branching constraint is removed and the LP basis is stale. Before the next probe, the parent's LP basis should be restored so each probe starts from the same point. This ensures consistent scoring across candidates. New columns added by a probe enter hors-base in the restored basis — most LP solvers handle this gracefully, but a targeted test should verify no basis mismatch crash occurs.

**State after all probes:** The LP solution is stale after the last probe, but this is harmless — the next `evaluate!` call on a child node re-runs full CG from scratch.

**Logging:** Each probe logs: candidate variable, direction (floor/ceil), CG iterations consumed, dual bound obtained, infeasibility status, and `sb_score`. This is essential for debugging and parameter tuning.

### Strategy type

```julia
struct StrongBranching <: AbstractBranchingStrategy
    max_candidates::Int       # default 5
    max_cg_iterations::Int    # default 10
    mu::Float64               # default 1/6
    rule::AbstractBranchingRule  # default MostFractionalRule()
end
```

`select_branching_variable` flow:
1. `find_fractional_variables` → all fractional vars.
2. `select_candidates(rule, candidates, max_candidates)` → top-k.
3. `run_sb_probe` for each candidate.
4. `sb_score` → pick best.
5. Return `(orig_var, value)` of winner.

### Tests

- Unit: `sb_score` — formula correctness, both feasible, one infeasible, both infeasible.
- Unit: `run_sb_probe` — small CG-capable fixture using GAP test infrastructure.
- Unit: `run_sb_probe` with both children infeasible — verify early detection.
- Unit: LP basis restoration after probe with new columns — verify no solver crash from basis mismatch.
- E2e: Small GAP with `StrongBranching()` → finds optimal, fewer nodes than `MostFractionalBranching`.

### Files

| Action | File |
|--------|------|
| Create | `src/BranchCutPrice/strong_branching.jl` |
| Modify | `src/BranchCutPrice/branching.jl` — add `add_branching_constraint!` and `remove_branching_constraint!` |
| Modify | `src/BranchCutPrice/branching_strategy.jl` — add `StrongBranching` + `select_branching_variable` |
| Modify | `src/BranchCutPrice/BranchCutPrice.jl` — include + export |

---

## Sub-issue 4: Pseudocosts and reliability branching

**Depends on:** Sub-issue 3.

### Pseudocost tracking

```julia
mutable struct PseudocostRecord
    sum_down::Float64    # Σ (Δ⁻ / frac_part)
    count_down::Int
    sum_up::Float64      # Σ (Δ⁺ / (1 - frac_part))
    count_up::Int
end

struct PseudocostTracker{X}
    records::Dict{X,PseudocostRecord}
    reliability_threshold::Int  # default 8
end
```

Unit pseudocosts normalize the dual bound improvement by fractionality, making scores comparable across variables with different fractional parts.

### Functions

- `update_pseudocosts!(tracker, candidate, result)` — updates sum/count for down and up. Skips infeasible sides.
- `estimate_score(tracker, candidate; mu=1/6)` — `score_down = mean_unit_down * frac_part`, `score_up = mean_unit_up * (1 - frac_part)`. Same product formula as `sb_score`.
- `is_reliable(tracker, candidate)` — `min(count_down, count_up) >= reliability_threshold`.

### Strategy type

```julia
struct ReliabilityBranching <: AbstractBranchingStrategy
    max_candidates::Int          # default 15
    max_cg_iterations::Int       # default 10
    mu::Float64                  # default 1/6
    reliability_threshold::Int   # default 8
    rule::AbstractBranchingRule  # default MostFractionalRule()
    pseudocosts::PseudocostTracker
end
```

`select_branching_variable` flow:
1. `find_fractional_variables` → all fractional vars.
2. `select_candidates(rule, candidates, max_candidates)` → top-k.
3. For each candidate: if `is_reliable` → `estimate_score`; else → `run_sb_probe` + `update_pseudocosts!` + `sb_score`.
4. Pick best score, return `(orig_var, value)`.

**Pseudocost updates from two sources:**
1. **During probes** (in `select_branching_variable`): after each `run_sb_probe`, call `update_pseudocosts!` with the `SBCandidateResult`. This covers unreliable variables.
2. **After node evaluation** via `on_node_evaluated(strategy, node_data, cg_output)` callback: once CG completes on a child node, compute Δ from the parent's LP obj and the child's dual bound, and call `update_pseudocosts!` for the variable that was branched on. This requires storing the branching variable and parent LP obj in `BPNodeData` at branching time. This is the primary source of observations once variables become reliable.

The `on_node_evaluated` interface is defined on `AbstractBranchingStrategy` with a no-op default. Only `ReliabilityBranching` overrides it. `evaluate!` calls `on_node_evaluated(space.branching_strategy, ...)` unconditionally — no `isa` check needed.

**Cold start:** Early in the tree, most variables are unreliable → behaves like full strong branching. As the tree grows, probes decrease and pseudocost estimates take over.

**Lifecycle:** `PseudocostTracker` lives inside `ReliabilityBranching`, which lives inside `BPSpace`. It accumulates across all nodes — pseudocosts are global properties of variables, not node-local.

### Tests

- Unit: `update_pseudocosts!` — cold start, multiple observations, skip infeasible.
- Unit: `estimate_score` — known values, verify formula.
- Unit: `is_reliable` — below threshold, at threshold, above.
- E2e: Small GAP with `ReliabilityBranching()` → finds optimal solution.
- E2e: Regression — `MostFractionalBranching` gives same result as before.

### Files

| Action | File |
|--------|------|
| Create | `src/BranchCutPrice/pseudocosts.jl` |
| Modify | `src/BranchCutPrice/branching_strategy.jl` — add `ReliabilityBranching` + `select_branching_variable` |
| Modify | `src/BranchCutPrice/bp_output.jl` — add `branching_var` and `parent_lp_obj` to `BPNodeData` |
| Modify | `src/BranchCutPrice/evaluator.jl` — call `update_pseudocosts!` after CG on child nodes |
| Modify | `src/BranchCutPrice/BranchCutPrice.jl` — include + export |

---

## Dependency graph

```
Sub-issue 1 (max_cg_iterations) ──┐
                                  ├──→ Sub-issue 3 (strong branching) ──→ Sub-issue 4 (reliability branching)
Sub-issue 2 (strategy interface) ─┘
```

Sub-issues 1 and 2 can be developed in parallel. Sub-issue 3 requires both. Sub-issue 4 requires 3.

---

## Future improvements (out of scope)

- **Early termination in probes:** Once pseudocosts are available (sub-issue 4), skip candidates whose estimated score is well below the best evaluated score.
- **Refactor `create_branching_children`** to use `add_branching_constraint!` / `remove_branching_constraint!` instead of going through `LocalCutTracker` — unify the branching constraint lifecycle.
- **Global iteration budget for probes:** Make `max_cg_iterations` apply as a total budget across all phases instead of per-phase, for more predictable probe cost.

---

## Reusable existing infrastructure

- `project_to_original(decomp, pool, master_primal_values)` — `src/Reformulation/interface.jl`
- `compute_branching_column_coefficient(decomp, orig_var, sp_id, sol)` — `src/Reformulation/interface.jl`
- `most_fractional_original_variable(ctx, primal_values)` — `src/BranchCutPrice/branching.jl`
- `create_branching_children(...)` — `src/BranchCutPrice/branching.jl`
- `MathOptState.apply_change!` for `AddLocalCutChange`/`RemoveLocalCutChange` — `src/MathOptState/local_cut_state.jl`
- `MathOptState.next_id!(cut_tracker)` — `src/MathOptState/local_cut_state.jl`
- `_rebuild_branching_constraints!(space)` — `src/BranchCutPrice/evaluator.jl`
- `ColGen.run_column_generation(ctx)` — `src/ColGen/context.jl`
