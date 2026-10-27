# Solving

Both solvers follow one pattern: build a config, build a workspace from the
decomposition and the config, run it.

```julia
output = run_column_generation(ColGenWorkspace(decomp, ColGenConfig()))
output = run_branch_and_price(BranchCutPriceWorkspace(decomp, BranchCutPriceConfig()))
```

## Column generation

### `ColGenConfig`

| Keyword | Default | Meaning |
|---------|---------|---------|
| `smoothing_alpha` | `0.0` | Wentges smoothing coefficient; `0.0` disables it |
| `max_cg_iterations` | `1000` | hard limit on iterations per phase |
| `max_misprice_iterations` | `10` | cap on misprice iterations per iteration |
| `silent` | `false` | skip logging |

### `ColGenOutput`

| Field | Meaning |
|-------|---------|
| `status` | a `ColGenStatus` |
| `master_lp_obj` | master LP value, or `nothing` |
| `incumbent_dual_bound` | best Lagrangian dual bound, or `nothing` |
| `ip_incumbent` | integer solution found, or `nothing` |

`ColGenStatus` is one of `optimal`, `master_infeasible`,
`subproblem_infeasible`, `iteration_limit`, `ip_pruned`.

When pricing finds no column but the dual bound and the master value still
differ, Vertigo warns that column generation converged with an open gap.

### Logging

`ColGenLoggerWorkspace(workspace; log_level = 1)` wraps a workspace; level 2
adds detail. `ColGenConfig(silent = true)` turns the log off.

## Branch-and-price

### `BranchCutPriceConfig`

| Keyword | Default | Meaning |
|---------|---------|---------|
| `colgen` | `ColGenConfig()` | configuration of the column generation at each node |
| `cg_log_level` | `0` | `> 0` wraps column generation in a logger of that level |
| `strategy` | `DepthFirstStrategy()` | tree search strategy |
| `node_limit` | `10_000` | maximum nodes to explore |
| `tol` | `1e-6` | numerical tolerance |
| `rmp_heuristic` | `true` | run the restricted master IP heuristic at each node |
| `rmp_time_limit` | `60.0` | seconds allowed to that heuristic per node |
| `separator` | `nothing` | robust cut separator |
| `max_cut_rounds` | `0` | cut rounds per node |
| `min_gap_improvement` | `0.01` | relative gap improvement needed to continue cut rounds |
| `branching_strategy` | `MostFractionalBranching()` | how to choose the branching variable |
| `log_level` | `0` | tree logger: `0` off, `1` table, `2` verbose |
| `dot_file` | `nothing` | path of a Graphviz `.dot` file of the tree |

Other tree strategies are `BreadthFirstStrategy` and
`DualBoundBestFirstStrategy`. `MultiPhaseStrongBranching` is the alternative
branching strategy.

### `BPOutput`

| Field | Meaning |
|-------|---------|
| `status` | `:optimal`, `:infeasible` or `:node_limit` |
| `incumbent` | best integer solution, or `nothing`; `incumbent.obj_value` is its cost |
| `best_dual_bound` | best dual bound over all nodes |
| `nodes_explored` | nodes evaluated |
| `root_lp_value` | master LP value at the root, or `nothing` |

> **Note.** Two defects affect these fields; see the [FAQ](../faq.md): the dual bound
> can be `-Inf` after the tree is exhausted (#75), and a node whose
> solution is feasible but with an open gap can end the search (#76).

### Cut separation

Implement `AbstractCutSeparator` (a `separate` method returning
`SeparatedCut`s), or wrap a callback in `CustomCutSeparator(callback)`, where
`callback(x::Dict{X,Float64})` returns a `Vector{SeparatedCut{X}}`. Pass it as
`separator` with `max_cut_rounds > 0`.
