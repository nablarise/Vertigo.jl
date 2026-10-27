# Vertigo.jl Introduction

!!! warning

    We assume that readers are familiar with integer programming and exact optimization methods.

    Vertigo is under active development. API may be subject to changes at any time.

Vertigo.jl is a Julia framework for branch-cut-and-price algorithms. The user writes a compact
JuMP model, declares which variables and constraints belong to the master problem and which to
the subproblems, and Vertigo.jl solves it by Dantzig-Wolfe decomposition, column generation and
branch-and-price.

Vertigo provides:

- a declarative way to describe the decomposition of a JuMP model
  (`@dantzig_wolfe`);
- column generation with state of the art acceleration methods;
- branch-and-price over a configurable tree search;
- user-overridable functions and callbacks allowing for problem-specific algorithm given by the
  user.

## Installation

Vertigo is not registered. Clone the repository and add it to your environment:

```julia
using Pkg
Pkg.develop(path = "path/to/Vertigo.jl")
```

You also need JuMP and a MIP solver for the subproblems, e.g. HiGHS:

```julia
Pkg.add(["JuMP", "HiGHS"])
```

## A first run

```julia
using JuMP, HiGHS, Vertigo

cost = [3.0 13.0 16.0 4.0 5.0 5.0 11.0 18.0;
        5.0 11.0 17.0 4.0 9.0 8.0 9.0 9.0]
weight = [6.0 5.0 3.0 1.0 2.0 2.0 2.0 2.0;
          3.0 2.0 2.0 2.0 1.0 4.0 6.0 6.0]
capacity = [13.0, 14.0]
K = 1:2; T = 1:8

model = Model(HiGHS.Optimizer)
set_silent(model)
@variable(model, x[k in K, t in T], Bin)
@constraint(model, assign[t in T], sum(x[k, t] for k in K) == 1)
@constraint(model, knapsack[k in K], sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
@objective(model, Min, sum(cost[k, t] * x[k, t] for k in K, t in T))

decomp, sp_map = @dantzig_wolfe model begin
    master() => (constraints = assign[_])
    subproblem(k) => (variables = x[k, _], constraints = knapsack[k])
end

output = run_column_generation(ColGenWorkspace(decomp, ColGenConfig()))
output.status                # optimal
output.incumbent_dual_bound  # 63.5 (the integer optimum is 65)
```

