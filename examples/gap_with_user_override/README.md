<!-- Copyright (c) 2026 Nablarise. All rights reserved. -->
<!-- Author: Guillaume Marques <guillaume@nablarise.com> -->
<!-- SPDX-License-Identifier: MIT -->

# GAP benchmark, user-overridden subproblems

The [`gap`](../gap) benchmark, with every machine's pricing subproblem
**solved by your own code** instead of an MOI model. It shows how to plug a
custom pricer into Vertigo and what changes when that pricer is heuristic.

## What differs from `gap`

| | `gap` | `gap_with_user_override` |
|---|---|---|
| Subproblem `k` | `subproblem(k)`: Vertigo builds a knapsack MOI model | `subproblem(k, user_overriden = true)`: no model |
| Knapsack constraint | modeled in the subproblem | not modeled; the pricer enforces it |
| Pricing | MIP solve | exact dynamic program, see `knapsack_pricer.jl` |
| Instances, optima | `gap/instances*`, `instances_metadata.json` | the same files, shared |

`gap_model.jl` builds the same flat JuMP model, decomposes it with

```julia
decomp, sp_map = @dantzig_wolfe model begin
    master() => (constraints = assign[_])
    subproblem(k, user_overriden = true) => (
        variables = x[k, _],
        constraints = knapsack[k]
    )
end
```

and installs a `KnapsackPricer` for each machine in `sp_models(decomp)`.
Vertigo writes the reduced task costs into the pricer through
`MOI.modify`, then calls `CG.optimize_pricing_problem!` on it.

## The pricer

`knapsack_pricer.jl` solves the machine's knapsack exactly, per pricing call:

1. `solve_knapsack` runs the textbook 0/1 knapsack DP over the capacity
   (`best[c]` = lowest reduced cost with a load of at most `c`), skipping tasks
   whose reduced cost is not negative, then reads the chosen tasks back;
2. the optimal task set becomes one column;
3. its value is reported as both the primal and the dual bound of the subproblem,
   since the DP is exact.

Weights and capacities must be integers (true of the GAPLIB instances); the pricer
raises an error otherwise.

## Checking it

Pricing is exact, so column generation must reach the same root LP value as the MIP
pricing of `gap`. On the smoke set the two agree on all seven instances
(`root_lp_value` in the result files).

## Running

From the repository root:

```
julia --project=examples examples/gap_with_user_override/run_quick.jl
julia --project=examples examples/gap_with_user_override/run_full.jl
```

Same smoke set, node caps and result files as `gap`, written under
`gap_with_user_override/results/`.
