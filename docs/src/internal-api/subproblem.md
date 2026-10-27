# Subproblem

A subproblem generates the columns of the master. For each subproblem, the
`DWReformulation` stores its data and its model, and column generation prices
it at every iteration.

## What it contains

`SubproblemData` (`src/Reformulation/subproblem_data.jl`) holds, for each
subproblem identified by a `PricingSubproblemId`:

| Field | Meaning |
|-------|---------|
| `variables` | the subproblem's variables, as `MOI.VariableIndex` |
| `original_costs` | the cost of each variable in the original objective |
| `coupling_coeffs` | the coefficient of each variable in each coupling constraint of the master |
| `fixed_cost` | the constant added to the cost of every column |
| `convexity_lb`, `convexity_ub` | bounds on the total weight of the subproblem's columns in the master |

Its model is `sp_model(decomp, sp_id)`: an MOI model built by the DML, or
`nothing` for an overridden subproblem until a pricer is installed.

## Mapping to the original model

`OneToOneMapping` links the original JuMP variables and the subproblem
variables in both directions:

- `mapped_subproblem_var(decomp, x[k, t])` returns `(sp_id, MOI.VariableIndex)`;
- `mapped_original_var(decomp, sp_id, index)` returns the original variable.

Columns are expressed over subproblem variables. The mapping is how branching
constraints and cuts, which are written over original variables, reach them.

## Reduced costs

At each iteration, after the master is solved, `compute_reduced_costs!`
(`src/ColGen/reduced_costs.jl`) computes the reduced cost of every subproblem
variable `j`:

```
reduced cost of j = original cost of j − Σ_i a_ij · π_i
```

where `π_i` are the duals of the coupling constraints and `a_ij` the coupling
coefficients, with an additional term for each branching constraint. The values
are written into the subproblem's model with `MOI.modify`, as coefficient
changes of the objective function. A pricer receives them the same way.

## Pricing

```text
                  +--------------------------------------------+
                  | sp_models(decomp)[sp_id]                   |
                  +--------------------------------------------+
                                         | (before pricing, update_reduced_costs! calls
                                         | MOI.modify on this stored object)
                                         v
                  +--------------------------------------------+
                  | PricingSubproblem(model)                   |
                  +--------------------------------------------+
                                         |
                                         v
                  +--------------------------------------------+
                  | get_pricing_subprob_optimizer(stage, sp)   |
                  +--------------------------------------------+
                                         |
                  +----------------------+--------------------+
                  | MOI model                   user pricer   |
                  v                                           v
 +----------------------------------+        +----------------------------------+
 | SubproblemMoiOptimizer           |        | the pricer itself (your method)  |
 +----------------------------------+        +----------------------------------+
                  |                                           |
                  v                                           v
 +----------------------------------+        +----------------------------------+
 | optimize_pricing_problem!:       |        | optimize_pricing_problem!:       |
 | MOI.optimize!                    |        | your method                      |
 +----------------------------------+        +----------------------------------+
                  |                                           |
                  +----------------------+--------------------+
                                         v
                  +--------------------------------------------+
                  | PricingSolution                            |
                  +--------------------------------------------+
```

`get_pricing_subprobs` wraps each model in a `PricingSubproblem`. The main loop
asks `get_pricing_subprob_optimizer` for the optimiser of the current stage;
for the `ExactStage` the default is `SubproblemMoiOptimizer`. It then calls
`optimize_pricing_problem!`, dispatching on the type of that optimiser:

- **Model solved by an optimiser**: sets the sense, calls `MOI.optimize!`,
  and turns the result into a `PricingSolution`. The column's reduced cost is
  the objective value plus the fixed cost minus the convexity duals.
- **Overridden subproblem**: the method you define for your pricer, see the
  [overriding tutorial](../tutorials/overriding-subproblems.md).

### The result of a pricing call

`PricingSolution` carries:

| Field | Meaning |
|-------|---------|
| `is_infeasible`, `is_unbounded` | termination information; infeasible ends the iteration, unbounded throws `UnboundedProblemError` |
| `primal_bound` | the reduced cost of the best column found |
| `dual_bound` | a lower bound on the subproblem optimum; it feeds the [dual bound](master.md) of the master |
| `primal_sols` | the columns found, each a `PricingPrimalSolution` with an `is_improving` flag |

A column is improving when its reduced cost is below `-RC_IMPROVING_TOL` (for a
minimisation problem). `GeneratedColumns` keeps only the improving ones, and
`insert_columns!` adds them to the master.

When the optimiser proves optimality, `primal_bound` and `dual_bound` agree.
For heuristic pricing they differ: the dual bound still has to be valid, and the
primal bound is used for stabilisation (the pseudo dual bound).
