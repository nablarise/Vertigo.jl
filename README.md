# Vertigo.jl

A Julia framework for branch-cut-and-price.

## Features

- **Declarative Dantzig-Wolfe decomposition** by describing which variables and constraints belong to the master and to each subproblem
- **Column generation**
- **Branch-and-price**
- **Tree search** with multiple strategies
- **Robust cut separation** with user-defined cut callbacks
- **Restricted master IP heuristic** for finding integer-feasible solutions

Built on [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) and [JuMP](https://github.com/jump-dev/JuMP.jl).

## Example

Generalized Assignment Problem solved with column generation:

```julia
using JuMP, HiGHS, Vertigo

# Instance data
cost    = [5.0 8.0 14.0 20.0 5.0 4.0 13.0;
           18.0 14.0 15.0 16.0 3.0 8.0 19.0]
weight  = [1.0 1.0 1.0 5.0 2.0 1.0 4.0;
           5.0 3.0 4.0 1.0 4.0 1.0 1.0]
capacity = [11.0, 14.0]
K = 1:2; T = 1:7

# Build JuMP model
model = Model(HiGHS.Optimizer)
set_silent(model)
@variable(model, x[k in K, t in T], Bin)
@constraint(model, assign[t in T], sum(x[k, t] for k in K) == 1)
@constraint(model, knapsack[k in K], sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
@objective(model, Min, sum(cost[k, t] * x[k, t] for k in K, t in T))

# Decompose: x and knapsack per machine, assign stays in master
decomp, sp_map = dantzig_wolfe_decomposition(
    model,
    MasterProblemUserDescription(constraints = assign),
    [
        SubproblemUserDescription(k; variables = x[k, :], constraints = knapsack[k]) for
        k in K
    ],
)

# Solve via column generation
config = ColGenConfig()
workspace = ColGenWorkspace(decomp, config)
ctx = ColGenLoggerWorkspace(workspace)
output = run_column_generation(ctx)

println("Status: ", output.status)           # optimal
println("Dual bound: ", output.incumbent_dual_bound)  # 63.0
```

## Tests

```julia
using Pkg; Pkg.test("Vertigo")
```

`test/VertigoUnitTests/` has one folder per module. `test/VertigoE2eTests/`
runs column generation and branch-and-price on small GAP instances, including
decompositions built with `dantzig_wolfe_decomposition`.

## Status

Experimental.
