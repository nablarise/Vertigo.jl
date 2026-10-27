# Branch-and-price

`run_branch_and_price` (`src/BranchCutPrice/`) embeds column generation in a
tree search. The search loop itself is generic (`src/TreeSearch/`); the
branch-and-price specifics are the search space, the node evaluator and the
branching.

## The search

`BranchCutPriceWorkspace` is a `TreeSearch.AbstractSearchSpace`. It wraps the
`ColGenWorkspace`, the domain and cut trackers (`MathOptState`), and the
incumbent and bound bookkeeping. It builds its `ColGenWorkspace` from the
decomposition and the `ColGenConfig` in its `BranchCutPriceConfig`.

`TreeSearch.search` (`src/TreeSearch/search_loop.jl`):

```text
+--------------------------------------------------------------+
| new_root, init_container                                     |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| stop, or no open node?                                       |<--+
| yes: return output(space) = BPOutput                         |   |
+--------------------------------------------------------------+   |
                                | no                               |
                                v                                  |
+--------------------------------------------------------------+   |
| select_node!                                                 |   |
+--------------------------------------------------------------+   |
                                |                                  |
                                v                                  |
+--------------------------------------------------------------+   |
| transition! to the node (not for the root)                   |   |
+--------------------------------------------------------------+   |
                                |                                  |
                                v                                  |
+--------------------------------------------------------------+   |
| evaluate! (BPEvaluator) returns the node status              |   |
+--------------------------------------------------------------+   |
                                |                                  |
                                v                                  |
+--------------------------------------------------------------+   |
| node status:                                                 |   |
|   BRANCH:   branch!, insert_children!                        |   |
|   FEASIBLE: on_feasible_solution!, prune!                    |   |
|   CUTOFF:   nothing                                          |   |
| then: next iteration                                         |---+
+--------------------------------------------------------------+
```

The strategy (`DepthFirstStrategy`, `BreadthFirstStrategy`, best-first,
beam, limited discrepancy) decides which open node `select_node!` returns.
`transition!` moves the master from the previous node to the next one through
`MathOptState`, which keeps the changes between nodes as forward and backward
differences.

## Evaluating a node

`evaluate!(::BPEvaluator, …)` (`src/BranchCutPrice/evaluator.jl`):

```text
+--------------------------------------------------------------+
| rebuild the branching constraints                            |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| run_column_generation                                        |<--+
+--------------------------------------------------------------+   |
                                |                                  |
                                v                                  |
+--------------------------------------------------------------+   |
| separate cuts                                                |   |
+--------------------------------------------------------------+   |
                                |                                  |
                                v                                  |
+--------------------------------------------------------------+   |
| another cut round?                                           |   |
| yes: run column generation again                             |---+
+--------------------------------------------------------------+
                                | no
                                v
+--------------------------------------------------------------+
| master or subproblem infeasible, or ip_pruned?               |
| yes: return CUTOFF                                           |
+--------------------------------------------------------------+
                                | no
                                v
+--------------------------------------------------------------+
| dual bound no better than the incumbent?                     |
| yes: return CUTOFF                                           |
+--------------------------------------------------------------+
                                | no
                                v
+--------------------------------------------------------------+
| column generation found a better integer solution?           |
| yes: return FEASIBLE                                         |
+--------------------------------------------------------------+
                                | no
                                v
+--------------------------------------------------------------+
| restricted master IP heuristic, if enabled                   |
+--------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------+
| return BRANCH                                                |
+--------------------------------------------------------------+
```

The heuristic may improve the incumbent, but the node is still branched.
Cut rounds stop according to `max_cut_rounds` and `min_gap_improvement`.

## Output

`TreeSearch.output(space)` builds the `BPOutput`: status, incumbent, best dual
bound, number of nodes and root LP value.
