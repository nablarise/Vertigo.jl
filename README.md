# Vertigo.jl

A Julia sandbox for branch-cut-and-price algorithms — and an experiment in AI-assisted code generation.

## Features

- **Declarative Dantzig-Wolfe decomposition** via the `@dantzig_wolfe` macro: pattern-based assignment of variables and constraints to master/subproblems
- **Column generation** with configurable iteration limits, dual bound tracking, and support for minimization and maximization
- **Wentges smoothing stabilization** for dual solutions during column generation
- **Branch-and-price** with pluggable branching strategies:
  - Most/least fractional branching rules
  - Multi-phase strong branching (LP probe, CG probe)
  - Pseudocost branching with persistence across the tree
- **Tree search** with multiple strategies: depth-first, breadth-first, best-first (dual bound), beam search, limited discrepancy
- **Robust cut separation** with user-defined cut callbacks
- **Restricted master IP heuristic** for finding integer-feasible solutions
- **MOI model state tracking** (column/cut/basis/bounds/integrality state) for efficient node transitions
- **Pure master variables** and fixed master costs in the decomposition
- Built on [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) and [JuMP](https://github.com/jump-dev/JuMP.jl)

## Example

Generalized Assignment Problem solved with column generation via `@dantzig_wolfe`:

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
decomp, sp_map = @dantzig_wolfe model begin
    x[k, _]      => subproblem(k)
    assign[_]     => master()
    knapsack[k]   => subproblem(k)
end

# Solve via column generation
ctx = ColGenLoggerContext(ColGenContext(decomp, ColumnPool()))
output = run_column_generation(ctx)

println("Status: ", output.status)           # optimal
println("Dual bound: ", output.incumbent_dual_bound)  # 63.0
```

## Status

Experimental — under active development.

Vertigo.jl is also an experiment in AI-assisted development. Most of the code in this repository was written collaboratively between a human and GenAI tools. Browse the git history for insight into what that workflow looks like in practice.
