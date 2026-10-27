# Tutorial: the generalised assignment problem

The generalised assignment problem (GAP) assigns each of `T` tasks to exactly
one of `K` machines. Machine `k` has a capacity, and assigning task `t` to
machine `k` costs `cost[k, t]` and uses `weight[k, t]` of that capacity. The
aim is to minimise the total cost.

This tutorial builds the model, decomposes it, solves the linear relaxation
by column generation, then solves the integer problem by branch-and-price.

## 1. The flat model

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
```

`x[k, t]` is 1 when task `t` runs on machine `k`. The `assign` constraints
link the machines, since each task must be covered once overall. Each
`knapsack[k]` involves only machine `k`'s variables. That is the block
structure to exploit.

Every variable and constraint must be **named**, as above: the rules below
refer to them by name.

## 2. The decomposition

```julia
decomp, sp_map = @dantzig_wolfe model begin
    master() => (constraints = assign[_])
    subproblem(k) => (variables = x[k, _], constraints = knapsack[k])
end
```

Each line says what a model contains: `model => (variables = ..., constraints = ...)`.

- `master() => (constraints = assign[_])`: the `assign` constraints stay in
  the master. `_` is a wildcard matching every task index, so all the `assign`
  constraints are located in the master problem.
- `subproblem(k) => (variables = x[k, _], constraints = knapsack[k])`:
  subproblem `k` holds the variables `x[k, t]` and machine `k`'s capacity
  constraint. The `k` in the slice descriptions is captured and reused as the
  subproblem identifier.

The macro returns the decomposition `decomp` and `sp_map`, a dictionary from your identifiers (`1`
and `2` in the GAP example) to the internal `PricingSubproblemId`s. The original `model` is not
modified.

## 3. Column generation

```julia
config = ColGenConfig()
workspace = ColGenWorkspace(decomp, config)
output = run_column_generation(workspace)

output.status                # optimal
output.master_lp_obj         # 63.5
output.incumbent_dual_bound  # 63.5
```

`status` is a `ColGenStatus`. `optimal` means column generation converged;
the master LP value and the dual bound agree at 63.5, the optimum of the
linear relaxation of the decomposed model. The optimum of the integer problem
is 65, so the relaxation is fractional: the master solution mixes columns and
no integer assignment reaches 63.5.

!!! note

    All solvers are silent by default. To see the iteration log, TODO.

## 4. Branch-and-price

Since the linear relaxation is fractional, column generation alone does not solve the problem.
The branch-and-price algorithm in Vertigo.jl branches on the original variables and runs column
generation at each node until it finds the integer optimum:

```julia
decomp, _ = @dantzig_wolfe model begin
    master() => (constraints = assign[_])
    subproblem(k) => (variables = x[k, _], constraints = knapsack[k])
end

config = BranchCutPriceConfig()
output = run_branch_and_price(BranchCutPriceWorkspace(decomp, config))

output.incumbent.obj_value   # 65.0
output.root_lp_value         # 63.5
output.nodes_explored        # 5
```

The root relaxation, 63.5, is a lower bound; the best integer solution found
costs 65.0, and the tree has 5 nodes.

## Where to go next

TODO

- [Decomposition rules](../user-api/decomposition.md) for slice descriptions, containers and options.
- [Solving](../user-api/solving.md) for every configuration field.
