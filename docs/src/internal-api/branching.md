# Module: Branching

!!! warning

    This page is experimental and has not been reviewed.

`Branching` (`src/BranchCutPrice/Branching/`) is a submodule of `BranchCutPrice`. It chooses the
original variable on which a node is branched; `BranchCutPrice` turns the choice into two children.
It uses the decomposition, column generation and `MathOptState`, and never `TreeSearch`.

## From the master solution to the children

```text
+----------------------------------------------------------------------------------------+
| TreeSearch.branch!(space, node)                                       (space.jl)       |
| primal_values = the master LP solution                                                 |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| select_branching_variable(strategy, space, node, primal_values)       (Branching)      |
| returns a BranchingResult: branching_ok, all_integral or node_infeasible,              |
| and for branching_ok the original variable and its value                               |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| create_branching_children                                       (BranchCutPrice)       |
| left child: x <= floor(value), right child: x >= ceil(value), as local cuts            |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| BPEvaluator, at the child                                                              |
| lists the active branching constraints in the ColGen workspace                         |
+----------------------------------------------------------------------------------------+
```

An original variable is a key of `project_to_original` (see the
[Decomposition](decomposition.md) module): the sum, over the columns, of the subproblem
variables mapped to it, each weighted by the value of its column. A branching decision is
one such variable. Its constraint has, for every column, the sum of those subproblem variables
as coefficient, and `ColGen` keeps it in `branching_constraints`. The dual of the constraint
enters the reduced cost of every subproblem variable mapped to the original variable
(`reduced_costs.jl`), and every new column gets its coefficient when it is inserted
(`column_insertion.jl`).

## Strategies

A strategy is a subtype of `AbstractBranchingStrategy` with a method of
`select_branching_variable`. `BranchCutPriceConfig.branching_strategy` holds it.

- `MostFractionalBranching` takes the original variable whose value is the closest to a half
  (`most_fractional_original_variable`, from `find_fractional_variables`).
- `MultiPhaseStrongBranching` probes candidates, as follows.

```text
+----------------------------------------------------------------------------------------+
| fractional candidates, ranked by their pseudocost estimate, at most max_candidates     |
| no parent LP value yet: the first candidate is taken                                   |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| for each phase, in order (LPProbePhase, then CGProbePhase by default)                  |<--+
|   for each candidate: probe both children, or reuse the pseudocost estimate when it    |   |
|   is reliable (CG phase only); score it with sb_score; stop the phase after            |   |
|   `lookahead` candidates without a better score                                        |   |
|   keep the best keep_fraction of the candidates for the next phase                     |---+
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| the best candidate of the last phase                                                   |
| a probe where both children are infeasible: the node is infeasible                     |
+----------------------------------------------------------------------------------------+
```

A probe (`run_sb_probe`) adds the branching constraint of a child to the master and to the
`ColGen` workspace, then either solves the master LP (`LPProbePhase`) or runs column generation
for at most `max_cg_iterations` iterations (`CGProbePhase`), and removes the constraint. The
iteration limit, the IP incumbent, the primal bound, the branching constraints and the LP basis are
saved before the first child and restored after each one.

The pseudocosts (`PseudocostTracker`) are updated from the probes and, passively, after every
evaluated node: `on_node_evaluated` reads the branching variable, the direction, the fraction and
the parent LP stored in `node.user_data`.

## Context and logging

A `BranchingContext` receives a call before and after each step of the selection
(`before_branching_selection`, `before_probe`, `after_probe`, `after_phase_filter`, ...).
`DefaultBranchingContext` does nothing; `BranchingLoggerContext` prints the strong-branching log.
`BranchCutPriceWorkspace` picks one from `log_level` and gives it to the strategy
(`with_branching_ctx`).

## Details worth knowing

- The accessors `bp_decomp`, `bp_pool`, `bp_branching_constraints`, ... (`interface.jl`) exist so
  the module works on a `ColGenWorkspace` and on a `ColGenLoggerWorkspace` alike.
- The strategies read `space.ws`, `space.backend` and `space.tolerance`, and `node.user_data.cg_output`,
  without naming `BranchCutPriceWorkspace`.
- A decision is one original variable. Branching on a sum of variables (an edge, the number of
  vehicles, a cutset) needs a way to map several subproblem variables to one original variable, or a
  change of the decision itself.

## Files

| File | Content |
|------|---------|
| `interface.jl` | `bp_*` accessors of the `ColGen` workspace |
| `branching_utils.jl` | `most_fractional_original_variable` |
| `branching_candidates.jl` | `BranchingCandidate`, `find_fractional_variables` |
| `branching_rules.jl` | `MostFractionalRule`, `LeastFractionalRule`, `select_candidates` |
| `branching_strategy.jl` | `BranchingResult`, `AbstractBranchingStrategy`, `MostFractionalBranching` |
| `phases.jl` | `LPProbePhase`, `CGProbePhase` |
| `strong_branching.jl` | branching constraint of a probe, `run_sb_probe`, `sb_score` |
| `lp_probe.jl`, `cg_probe.jl` | `probe_candidate` of each phase |
| `pseudocosts.jl` | `PseudocostTracker`, `estimate_score`, `is_reliable` |
| `kernel.jl` | selection kernel, `MultiPhaseStrongBranching`, `on_node_evaluated` |
| `context.jl`, `logger.jl` | `BranchingContext`, `DefaultBranchingContext`, `BranchingLoggerContext` |
