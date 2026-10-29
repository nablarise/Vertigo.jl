# FAQ

!!! warning

    This page is experimental and has not been reviewed.

## Known limitations

### Why is the dual bound `-Inf` and the status `:node_limit` after an optimal run?

Issue #75. When the tree is exhausted, `best_dual_bound` is not updated, so
it can stay at `-Inf`, and `status` can be `:node_limit` although the node
limit was not reached. The incumbent is still the best solution found. The
GAP in the [tutorial](tutorials/gap.md) shows it: it reports `:node_limit`
after 5 nodes, and `best_dual_bound` was `-Inf` in some runs and 63.5 in
others. Check `incumbent` and `root_lp_value` instead.

### Why did the search stop with an open gap?

Issue #76. A node whose solution is `FEASIBLE` while a gap remains ends the
search instead of continuing. The examples reproducing both bugs are on the
branch `debug/buggedexamples`.

### Why does `examples/gap/runner.jl` fail?

It calls `BranchCutPriceContext`, which was renamed
`BranchCutPriceWorkspace`. Build the workspace with
`BranchCutPriceWorkspace(decomp, BranchCutPriceConfig(...))`.

## Using Vertigo

### Which variables and constraints go to the master?

Constraints that couple the blocks (`assign` in the GAP). Variables of a block,
and the constraints involving only that block, go to the subproblem.

### Why do I get "is not assigned to the master or to a subproblem"?

Every variable and constraint of the model must be listed in exactly one
description, the master's or a subproblem's; see [Errors](user-api/errors.md).

### Can a constraint involve variables of two subproblems?

No; it belongs in the master. Vertigo rejects it with an error.

### Do I need to set an optimiser on the master and subproblems?

No. They use the optimiser of the model you decompose
(`Model(HiGHS.Optimizer)` in the examples). Models are silent by default; use
`SubproblemOptions(silent = false)` or `MasterProblemOptions(silent = false)` to see a solver's output.

### Can I solve a subproblem with my own algorithm?

Yes: describe it with `SubproblemOptions(user_overridden = true)`. See the
[tutorial](tutorials/overriding-subproblems.md) and
[Decomposition rules](user-api/decomposition.md).

## Conventions

### Which spelling does the documentation use?

British English: *generalised*, *optimiser*, *modelling*. Identifiers keep the
spelling of the code (`user_overridden`).
