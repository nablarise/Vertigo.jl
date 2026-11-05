# Module: BranchCutPrice

!!! warning

    This page is experimental and has not been reviewed.

The BranchCutPrice module (BCP, `src/BranchCutPrice/`) embeds column generation in a tree search:
`run_branch_and_price` solves the problem of a `DWDecomposition` to integrality. The search loop
itself is generic (`src/TreeSearch/`); the branch-and-price specifics are the search space, the node
evaluator and the branching, which this module provides.

## From the decomposition to the output

```text
+= the user's script ========================================================================+
|                                                                                            |
| config = BranchCutPriceConfig(...)                                              config.jl  |
| ws     = BranchCutPriceWorkspace(decomp, config)                                space.jl   |
| output = run_branch_and_price(ws)                                                space.jl  |
|                                                                                            |
+============================================================================================+
                                     |
                                     v
+= space.jl: run_branch_and_price(ws) =======================================================+
|                                                                                            |
| set the deadline of the search and of column generation                                    |
| pick the context: Graphviz dot logger, tree logger or none                                 |
| TreeSearch.search(strategy, ws, BPEvaluator())                                             |
|                                                                                            |
+============================================================================================+
                                     |
                                     v
+= TreeSearch.search (src/TreeSearch/search_loop.jl) ========================================+
|                                                                                            |
|    +- new_root, init_container ----------------------------------------------------------+ |
|    | the root node, its dual bound in the open nodes                                     | |
|    +-------------------------------------------------------------------------------------+ |
|       |                                                                                    |
|       v                                                                                    |
|    +- stop (node limit, time limit), or no open node? -----------------------------------+ |
|    | yes: leave the loop and build the output                                            | |
|    +-------------------------------------------------------------------------------------+ |
|       |                                                                                    |
|       v                                                                                    |
|    +- select_node!, then transition! to the node ----------------------------------------+ |
|    | MathOptState applies the differences between the two nodes                          | |
|    +-------------------------------------------------------------------------------------+ |
|       |                                                                                    |
|       v                                                                                    |
|    +- evaluate!(BPEvaluator, ws, node)  returns the node status -------------------------+ |
|    | see the next chart                                                                  | |
|    +-------------------------------------------------------------------------------------+ |
|       |                                                                                    |
|       v                                                                                    |
|    +- node status -----------------------------------------------------------------------+ |
|    | BRANCH:   branch!, insert_children!        space.jl, branching.jl                   | |
|    | FEASIBLE: on_feasible_solution!, prune!                                             | |
|    | CUTOFF:   nothing; the node is closed                                               | |
|    | then the next iteration                                                             | |
|    +-------------------------------------------------------------------------------------+ |
|                                                                                            |
+============================================================================================+
                                     |
                                     |  no open node, or stop
                                     v
+= bp_output.jl: output(ws) =================================================================+
|                                                                                            |
| BPOutput: status, incumbent, best dual bound, number of nodes, root LP value               |
|                                                                                            |
+============================================================================================+
```

The [Branching page](branching.md) follows `branch!` from the choice of the variable on.

Evaluating a node, the box `evaluate!` of the chart above:

```text
+= evaluator.jl: evaluate!(BPEvaluator, ws, node) ===========================================+
|                                                                                            |
|    +- evaluator.jl ----------------------------------------------------------------------+ |
|    | _set_incumbent_bound!, then _rebuild_branching_constraints!                         | |
|    +-------------------------------------------------------------------------------------+ |
|       |                                                                                    |
|       v                                                                                    |
|    +- Cutting/ --------------------------------------------------------------------------+ |
|    | run_column_generation, then _separate_and_add_cuts!                                 | |
|    | another round while stop_cutcolgen is false                                         | |
|    +-------------------------------------------------------------------------------------+ |
|       |                                                                                    |
|       v                                                                                    |
|    +- evaluator.jl: the node is closed? -------------------------------------------------+ |
|    | master or subproblem infeasible, ip_pruned, dual bound no better than               | |
|    | the incumbent, or time ran out in column generation: return CUTOFF                  | |
|    +-------------------------------------------------------------------------------------+ |
|       |                                                                                    |
|       v                                                                                    |
|    +- evaluator.jl: column generation found a better integer solution? ------------------+ |
|    | yes: store it as the incumbent, return FEASIBLE                                     | |
|    +-------------------------------------------------------------------------------------+ |
|       |                                                                                    |
|       v                                                                                    |
|    +- rmp_heuristic.jl ------------------------------------------------------------------+ |
|    | solve_restricted_master_ip!, if enabled; may improve the incumbent                  | |
|    +-------------------------------------------------------------------------------------+ |
|       |                                                                                    |
|       v                                                                                    |
|       return BRANCH                                                                        |
|                                                                                            |
+============================================================================================+
```

## Steps

!!! warning

    This section is to be reviewed.

The charts above, in words.

1. **Configuring** (`config.jl`). `BranchCutPriceConfig` is a plain data holder with a default for
   every field: the column generation configuration (`cg_config`), the search (`node_select_strategy`,
   `node_limit`, `time_limit`, `tolerance`), the cuts (`cut_config`, a `CutSeparationConfig` with `separator`, `max_cut_rounds` and
   `min_gap_improvement`), the restricted master heuristic (`rmp_use_heuristic`, `rmp_heuristic_time_limit`), the
   branching (`branching_strategy`) and the logs (`log_level`, `dot_file`; the column generation log is `cg_config.log_level`).
2. **Building the workspace** (`space.jl`). `BranchCutPriceWorkspace(decomp, config)` is the search
   space, a `TreeSearch.AbstractSearchSpace`. It builds the `ColGenWorkspace` from the
   decomposition and `config.cg_config` (wrapped in a logger when its `log_level > 0`), registers the
   domain and the cut trackers of `MathOptState` on the master, and copies the other fields of the
   configuration. It also holds the incumbent, the global dual bound and the dual bound of each
   open node.
3. **Searching** (`space.jl`, `src/TreeSearch/search_loop.jl`). `run_branch_and_price` sets the
   deadline of the search and of column generation, then calls `TreeSearch.search` with the
   strategy of the configuration (depth first by default); the dot logger or the tree logger wrap
   the workspace when they are enabled. The strategy decides which open node `select_node!`
   returns. `transition!` moves the master from the previous node to the next one: `MathOptState`
   keeps the changes between nodes as forward and backward differences, so that no node rebuilds
   the master. The search stops when `stop` is true (`node_limit` nodes evaluated, or the deadline
   passed) or when no open node is left.
4. **Evaluating a node** (`evaluator.jl`, `Cutting/`,
   `rmp_heuristic.jl`). `evaluate!` gives the master the incumbent value as primal bound, rebuilds
   the branching constraints of the node, and runs column generation. If a separator is set, the
   cut-and-column-generation loop adds the separated robust cuts and runs column generation again
   until `stop_cutcolgen` is true: `max_cut_rounds` rounds, no new cut, a column generation that is
   not optimal, or a relative improvement of the master LP below `min_gap_improvement`. Then the
   node is closed (`CUTOFF`) if the master or a subproblem is infeasible, if column generation
   proved the node cannot improve the incumbent (`ip_pruned`), if its dual bound is no better than
   the incumbent, or if the time ran out in column generation. If column generation found a better
   integer solution, it becomes the incumbent and the status is `FEASIBLE`. Otherwise the
   restricted master heuristic may improve the incumbent, and the status is `BRANCH`: the
   heuristic does not close the node.
5. **Branching** (`space.jl`, `branching.jl`). For a `BRANCH` node, `branch!` reads the primal
   solution of the master and asks the branching strategy for an original variable
   (`select_branching_variable`). If all the original variables are integral, or the node is
   infeasible, it has no child. Otherwise `create_branching_children` builds two children, each
   with a robust branching constraint on the columns, `≤ ⌊x⌋` and `≥ ⌈x⌉`, added as a `LocalCut`
   through a difference of `MathOptState`. The children enter the open nodes with the dual bound of
   their parent.
6. **Output** (`bp_output.jl`, `space.jl`). `TreeSearch.output` builds the `BPOutput`: the status
   (`:optimal` when the incumbent and the global dual bound are within `tolerance`, `:node_limit` or
   `:time_limit` when the search stopped before, `:infeasible` when it ended with no incumbent),
   the incumbent, the best dual bound, the number of nodes and the root LP value.

## Files

| File | Role |
|------|------|
| `BranchCutPrice.jl` | the module: it includes the files below and the `Branching` and `Cutting` modules, and exports the user-facing names |
| `config.jl` | `BranchCutPriceConfig`, the user-defined parameters |
| `space.jl` | `BranchCutPriceWorkspace`, the search space, its `TreeSearch` interface (`new_root`, `stop`, `transition!`, `branch!`, `output`, the logger protocol) and `run_branch_and_price` |
| `evaluator.jl` | `BPEvaluator` and `evaluate!`, with the pruning and the incumbent bookkeeping |
| `rmp_heuristic.jl` | `solve_restricted_master_ip!`, the restricted master heuristic |
| `branching.jl` | `create_branching_children`, the two children of a branching |
| `bp_output.jl` | `BPOutput`, the result of a run |
| `Cutting/` | the `Cutting` module: the separator interface (`AbstractCutSeparator`, `SeparatedCut`, `CustomCutSeparator`), `CutSeparationConfig` with `stop_cutcolgen`, the stopping rule of the cut-and-column-generation loop, and `_separate_and_add_cuts!`, which projects the master solution to the original variables, separates the robust cuts and adds them to the master |
| `logger.jl` | `BPDotLoggerContext`, which runs the search and writes the Graphviz `.dot` file of the tree |
| `Branching/` | the `Branching` module: the branching strategies and candidates; see the [Branching page](branching.md) |
