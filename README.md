# Vertigo.jl

A Julia framework for branch-cut-and-price, built with autonomous AI agents.

## Features

- **Declarative Dantzig-Wolfe decomposition** via the `@dantzig_wolfe` macro: pattern-based assignment of variables and constraints to master/subproblems
- **Column generation**
- **Branch-and-price**
- **Tree search** with multiple strategies
- **Robust cut separation** with user-defined cut callbacks
- **Restricted master IP heuristic** for finding integer-feasible solutions

Built on [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) and [JuMP](https://github.com/jump-dev/JuMP.jl).

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
pool = ColumnPool()
no_art = Dict{TaggedCI,Tuple{MOI.VariableIndex,MOI.VariableIndex}}()
no_leq = Dict{TaggedCI,MOI.VariableIndex}()
no_geq = Dict{TaggedCI,MOI.VariableIndex}()
ctx = ColGenLoggerContext(
    ColGenContext(decomp, pool, no_art, no_leq, no_geq)
)
output = run_column_generation(ctx)

println("Status: ", output.status)           # optimal
println("Dual bound: ", output.incumbent_dual_bound)  # 63.0
```

## Agent-driven development

Development is driven by autonomous AI agents running
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) in Docker
containers. The agents automate the full GitHub workflow — from issue to
merged PR.

Agents never auto-merge. A human reviews and merges approved PRs.

## Status

Experimental — under active development.
