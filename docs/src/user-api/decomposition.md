# Decomposition rules

!!! warning

    This page is experimental and has not been reviewed.

`@dantzig_wolfe` turns a JuMP model into a `DWReformulation`. You list what the
master and each subproblem contain:

```julia
decomp, sp_map = @dantzig_wolfe model begin
    master() => (constraints = assign[_])
    subproblem(k) => (variables = x[k, _], constraints = knapsack[k])
end
```

It returns `(decomp::DWReformulation, sp_map::Dict)`. The model is read, not
modified. The rules are checked when the macro runs: see [Errors](errors.md).

## Models

Each line has the form `model => (variables = ..., constraints = ...)`.

| Model | Meaning |
|-------|---------|
| `subproblem(id)` | the subproblem identified by `id`, any expression of the captured indices |
| `master()` | the master problem; written exactly once |

Each assignment, `variables` and `constraints`, takes one slice description or a vector of
slice descriptions, for example `variables = [x[k, _], y[k]]`. Both assignments are
optional, and they only help the reader: a slice description is assigned to the model
whichever assignment it is in.

## Slice descriptions

A slice description names a variable or constraint container and gives one symbol per
index.

| Slice description | Meaning |
|---------|---------|
| `x[k, t]` | every `x[k, t]`; `k` and `t` can be used in `subproblem(...)` |
| `x[k, _]` | every `x[k, t]`; `_` is a wildcard matching every value of the second index |
| `z` | a scalar variable or constraint named `z` |

Indices must be plain symbols or `_`. Each variable or constraint must be
matched by **exactly one** slice description, so two slice descriptions for the same name and
number of indices are rejected.

## Options

`master()` and `subproblem(id)` accept options, written as keywords:

| Option | Applies to | Effect |
|--------|------------|--------|
| `silent=true\|false` | master, subproblem | silence the solver of that model; unset means silent |
| `user_overridden=true` | subproblem | the subproblem is solved by your own algorithm; no MOI model is built |

```julia
subproblem(k, silent = false) => (variables = x[k, _], constraints = knapsack[k])
```

Lines assigning to the same model must not contradict each other: one line
may say `silent=false` and another leave it unset, but not `true` and
`false`. The latter fails with *Conflicting `silent` settings for subproblem 2:
all rules assigning to it must agree.*

### User-overridden subproblems

With `user_overridden=true`, the constraints assigned to the subproblem are not
modelled, since your algorithm enforces them. The variables still get a
`MOI.VariableIndex`, available through `mapped_subproblem_var`, and
`sp_models(decomp)[sp_id]` is `nothing` until you install your pricer there.
`ColGenWorkspace(decomp, config)` fails with a "has no pricer" error if you
forget.

## JuMP containers

Slice descriptions address elements by the indices you declared. This works for `Array`,
`DenseAxisArray`, `SparseAxisArray` and `Dict` containers, so
`x[k in ["a", "b"], t in 3:5]` is addressed by `("a", 3)`, not by position.
Scalars and containers can be mixed.

## Pure master variables

A variable assigned to `master()` is a *pure master variable*. Its cost,
bounds and integrality are kept (a binary variable stays in `[0, 1]`).
Query them with `pure_master_variables`, `pure_master_cost` and
`pure_master_bounds`.

## Optimisers

The master and subproblem models use the optimiser of the model you pass in.
To choose another, set it on the model before decomposing, with `set_optimizer(model, ...)`.

## Reading a decomposition

| Function | Returns |
|----------|---------|
| `subproblem_ids(decomp)` | the `PricingSubproblemId`s |
| `subproblem_variables(decomp, id)` | the variables of one subproblem |
| `coupling_constraints(decomp)` | the master constraints linking subproblems |
| `master_model(decomp)`, `sp_models(decomp)` | the MOI models |
| `is_minimization(decomp)` | the sense |
