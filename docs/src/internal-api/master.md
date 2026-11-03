# Master problem

The master problem chooses a combination of columns, each a feasible solution
of a subproblem, so that the coupling constraints are satisfied at minimum
cost. It is a linear program in MathOptInterface that grows during column
generation.

## What it contains

The `DWDecomposition` owns the MOI master model, `master_model(decomp)`. It
holds:

| Element | Source in the code |
|---------|--------------------|
| **Columns**: one variable `λ ≥ 0` per column found | added by `_insert_column!` during column generation |
| **Coupling constraints**: the constraints assigned to `master()` that involve subproblem variables | `coupling_constraints(decomp)`, as `TaggedCI`s |
| **Convexity constraints**: bound the total weight of the columns of each subproblem | `convexity_lb_ci` and `convexity_ub_ci`, with the bounds `convexity_bounds(decomp, sp_id)` |
| **Pure master variables**: the variables assigned to `master()` | `PureMasterVariableData`: id, cost, bounds, integrality and coupling coefficients |
| **Branching constraints** | `ColGenWorkspace.branching_constraints` (`ActiveBranchingConstraint`) |
| **Robust cuts** | `ColGenWorkspace.robust_cuts` (`ActiveRobustCut`) |
| **Artificial variables**, which keep the master feasible while few columns exist | `eq_art_vars`, `leq_art_vars`, `geq_art_vars` of the workspace; see [Phases](column-generation.md#Phases) |

Constraints are identified by `TaggedCI`, a typed constraint index that can be
stored in dictionaries and sorted. Solutions are expressed over
`MOI.VariableIndex`.

The `Master` wrapper in `ColGen` (`src/ColGen/context.jl`) bundles the MOI
model with the convexity constraints, the artificial variables and the ids of
the coupling constraints.

## Solving the restricted master

`optimize_master_lp_problem!` calls `MOI.optimize!` on the master and returns a
`MasterSolution`: the MOI termination, primal and dual statuses, a
`MasterPrimalSolution` and a `MasterDualSolution`. The dual solution keeps the
ids of the coupling constraints so that reduced costs can be computed with a
merge of two sorted vectors.

An infeasible master ends the iteration with a `master_infeasible` status; an
unbounded master throws `UnboundedProblemError`.

## Inserting a column

`_insert_column!` adds one column to the master. Its coefficients are:

- the **coupling coefficients** of the subproblem solution;
- one coefficient per active **branching constraint**, computed on the original
  variable it branches on;
- one coefficient per active **robust cut**, the sum of the cut's coefficients
  over the original variables of the solution;
- `1.0` in the **convexity constraints** of the column's subproblem.

The objective coefficient is the column's original cost, except in phase 1,
where it is zero so that only the artificial variables carry cost.

Every column is recorded in the `ColumnPool`, which is indexed by column
variable, by subproblem and by a fingerprint. A column already in the pool is
not inserted again.

## The dual bound

At the end of each pricing round, `compute_dual_bound` (`src/ColGen/dual_bounds.jl`)
combines the master dual solution with the dual bounds returned by the
pricing subproblems:

```
dual bound = master dual objective
             − Σ_k convexity duals × convexity bounds
             − pure master variables' bound duals × bounds
             + Σ_k (subproblem dual bound + fixed cost) × multiplicity
             + pure master variables' contribution
```

The multiplicity of a subproblem is its upper convexity bound when the
subproblem's contribution points in the improving direction, and its lower
bound otherwise. A subproblem's dual bound is the one its pricer returns, so a
heuristic pricer that returns an invalid bound makes this bound invalid.

Before using the master duals, the function recomputes the dual objective from
the constraint duals and the right-hand sides, and stops with a *Dual cost
mismatch* error if it differs from the one reported by the solver.
