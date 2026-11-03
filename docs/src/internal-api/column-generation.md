# Column generation

`run_column_generation` (`src/ColGen/`) solves the linear relaxation of the
decomposed problem. It repeats an iteration over phases and stages until no
column improves or a limit (`max_cg_iterations`) is reached.

> **Note.** `src/ColGen/coluna.jl` is a kernel marked "DO NOT MODIFY". The rest
> of `ColGen` implements its dispatch points.

## One iteration

`run_colgen_iteration!` (`src/ColGen/coluna.jl`):

```text
+--------------------------------------------------------------+
| optimize_master_lp_problem!                                  |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| master infeasible?                                           |
| yes: return (status master_infeasible)                       |
+--------------------------------------------------------------+
                                | no
                                v
+--------------------------------------------------------------+
| check_primal_ip_feasibility!                                 |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| dual solution: compute_reduced_costs!, update_reduced_costs! |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| stabilisation: smoothed duals                                |<--+
+--------------------------------------------------------------+   |
                                |                                  |
                                v                                  |
+--------------------------------------------------------------+   |
| for each subproblem:                                         |   |
| get_pricing_subprob_optimizer, optimize_pricing_problem!     |   |
+--------------------------------------------------------------+   |
                                |                                  |
                                v                                  |
+--------------------------------------------------------------+   |
| subproblem infeasible?                                       |   |
| yes: return (status subproblem_infeasible)                   |   |
+--------------------------------------------------------------+   |
                                | no                               |
                                v                                  |
+--------------------------------------------------------------+   |
| keep improving columns and the subproblem dual bounds        |   |
+--------------------------------------------------------------+   |
                                |                                  |
                                v                                  |
+--------------------------------------------------------------+   |
| compute_dual_bound                                           |   |
+--------------------------------------------------------------+   |
                                |                                  |
                                v                                  |
+--------------------------------------------------------------+   |
| misprice? (smoothing active, no improving column)            |   |
| yes: update_stabilization_after_misprice!, price again       |---+
+--------------------------------------------------------------+
                                | no
                                v
+--------------------------------------------------------------+
| insert_columns!                                              |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| iteration output                                             |
+--------------------------------------------------------------+
```

The pieces are described on the [master problem](master.md) page (solving,
duals, column insertion, dual bound) and the [subproblem](subproblem.md) page
(reduced costs, pricing).

## Phases

Column generation runs in phases (`CGPhase`), which differ in the master
objective:

| Phase | Master objective |
|-------|------------------|
| `Phase0` | the original objective plus artificial variables priced at `artificial_var_cost` and `convexity_artificial_var_cost` (10000.0 by default) |
| `Phase1` | the sum of the artificial variables only, to find a feasible master (columns get cost zero) |
| `Phase2` | the original objective, with no artificial variables |

## Stabilisation

With `smoothing_alpha > 0`, pricing uses Wentges-smoothed duals
(`WentgesSmoothing`, `src/ColGen/dw_stabilization.jl`), a mix of a stability
center and the current master duals. A *misprice* happens when smoothing is
active and the smoothed duals produce no improving column. The center is then
updated and the subproblems are priced again, up to `max_misprice_iterations`
times per iteration. Without smoothing the pricing loop runs once.

## Status of a run

The outcome is a `ColGenStatus`: `optimal`, `master_infeasible`,
`subproblem_infeasible`, `iteration_limit` or `ip_pruned`. When pricing finds no
column but the dual bound and the master value still differ, a warning is
logged: the pricing is then not exact.
