<!-- Copyright (c) 2025 Nablarise. All rights reserved. -->

# Branching Logger Context — Design Spec

## Problem

The multi-phase strong branching kernel (`kernel.jl`, `strong_branching.jl`)
embeds logging directly: `println`, `@warn`, `@debug`, formatting helpers,
and a `log_level::Int` parameter threaded through the kernel. This violates
the project's kernel pattern where kernels read as pseudocode with zero
logging (as established by `ColGen/coluna.jl` + `ColGen/logger.jl`).

## Solution

Extract all logging from the branching kernel into a `BranchingLoggerContext`
that wraps the default `BranchingContext`, following the same pattern as
`ColGenLoggerContext` wrapping `ColGenContext`.

## Design

### Context types

- **`BranchingContext`** — default context. All hook functions are no-ops.
- **`BranchingLoggerContext`** — wraps `BranchingContext`. Overrides hooks
  to emit formatted terminal output. Holds `io::IO`, `log_level::Int`,
  `t0::Float64`.

### Hook functions

The kernel calls these at well-defined points. Default dispatch on
`BranchingContext` is a no-op. `BranchingLoggerContext` overrides them.

| Hook | Call site | Arguments (besides context) |
|------|-----------|----------------------------|
| `before_branching_selection` | Start of `run_branching_selection` | candidates, phases |
| `before_probe` | Before probing one direction in `run_sb_probe` | phase, candidate, direction (`:left`/`:right`) |
| `after_probe` | After probing one direction in `run_sb_probe` | phase, candidate, direction, `SBProbeResult` |
| `after_candidate_eval` | After scoring in `_eval_candidate` | phase, idx, candidate, score, `SBCandidateResult` or `:reliable` |
| `on_both_infeasible` | Both branches infeasible in `_eval_candidate` | phase, idx, candidate |
| `after_phase_filter` | Between phases in `run_branching_selection` | phase label, before count, after count |
| `after_branching_selection` | Final pick in `run_branching_selection` | best candidate, best score |

### Kernel changes

- `run_branching_selection` gains a `context::BranchingContext` argument,
  loses `log_level::Int`.
- `_eval_candidate` loses `log::Bool` and `t0::Float64`, gains `context`.
- `run_sb_probe` gains `context` for `before_probe`/`after_probe` calls.
- All `println`, `@warn`, `@debug` removed from kernel and probe code.
- `_sb_log_header`, `_sb_log_candidate`, `_sb_log_selected` deleted from
  `strong_branching.jl`.
- `_sb_delta`: remove `@warn`, just return `0.0` when no dual bound.
- `space.jl`: remove `@debug` from `branch!`.

### Log levels in `BranchingLoggerContext`

- `log_level == 1`: summary per candidate (`after_candidate_eval`,
  `after_phase_filter`, `after_branching_selection`).
- `log_level >= 2`: per-direction probe details (`before_probe`,
  `after_probe`).

### File layout

- `src/Branching/context.jl` — `BranchingContext`, hook stubs (no-ops)
- `src/Branching/logger.jl` — `BranchingLoggerContext`, hook overrides
- `src/Branching/kernel.jl` — clean kernel, calls hooks on context
- `src/Branching/strong_branching.jl` — clean probing, calls probe hooks

Include order in `Branching.jl`: `context.jl` before `kernel.jl`,
`logger.jl` after `kernel.jl`.

### Wiring

`MultiPhaseStrongBranching` holds a `context::BranchingContext` field
(default `BranchingContext()`). `BPSpace` constructs the appropriate
context type based on its `log_level` and passes it to the strategy.
`select_branching_variable` forwards the context to
`run_branching_selection`.

### Output format

`BranchingLoggerContext` reproduces the same formatted output that exists
today (the `@sprintf` lines from the deleted helpers), keeping the output
stable for users.
