# Decomposition rules

!!! warning

    This page is experimental and has not been reviewed.

`dantzig_wolfe_decomposition` turns a JuMP model into a `DWReformulation`. You describe what the
master and each subproblem contain, with the JuMP variables and constraints themselves:

```julia
decomp, sp_map = dantzig_wolfe_decomposition(
    model,
    MasterProblemUserDescription(constraints = assign),
    [
        SubproblemUserDescription(k; variables = x[k, :], constraints = knapsack[k]) for
        k in K
    ],
)
```

It returns `(decomp::DWReformulation, sp_map::Dict)`. The model is read, not
modified. The descriptions are checked before anything is built: see [Errors](errors.md).

## Descriptions

| Description | Meaning |
|-------------|---------|
| `SubproblemUserDescription(id; variables, constraints, options)` | the subproblem identified by `id` |
| `MasterProblemUserDescription(variables, constraints, options)` | the master problem |

The master is described once, and the subproblems by a vector with one description each. The
identifier of a subproblem is any value that can be sorted together with the others: an integer,
a string, a tuple such as `(m, t)`. It is the key of `sp_map`. Two subproblems cannot have the
same identifier.

`variables` and `constraints` are optional and take what the description owns:

- a variable or constraint reference, such as `knapsack[1]`;
- a container of references, such as `assign`, or a slice such as `x[k, :]`;
- a vector, a tuple or a dictionary of these, nested if needed:
  `constraints = [assign, max_unassign]`.

Each variable and each constraint of the model must be owned by **exactly one** description.
`variables` takes variables and `constraints` takes constraints; giving one in place of the other
is an error. A variable or constraint does not need a name, since it is found by its reference.

## Options

The options of a model are held by a structure, passed as the `options` argument of its
description:

| Structure | Field | Effect |
|-----------|-------|--------|
| `MasterProblemOptions` | `silent = true` | `false` shows the output of the solver of the master |
| `SubproblemOptions` | `silent = true` | `false` shows the output of the solver of the subproblem |
| `SubproblemOptions` | `user_overridden = false` | `true`: the subproblem is solved by your own algorithm; no MOI model is built |

```julia
SubproblemUserDescription(
    k;
    variables = x[k, :],
    constraints = knapsack[k],
    options = SubproblemOptions(silent = false),
)
```

The options of a subproblem are ordinary values, so they can depend on its identifier:
`options = SubproblemOptions(user_overridden = (k == 1))` overrides the subproblem 1 only.

### User-overridden subproblems

With `user_overridden = true`, the constraints assigned to the subproblem are not
modelled, since your algorithm enforces them. The variables still get a
`MOI.VariableIndex`, available through `mapped_subproblem_var`, and
`sp_models(decomp)[sp_id]` is `nothing` until you install your pricer there.
`ColGenWorkspace(decomp, config)` fails with a "has no pricer" error if you
forget.

## JuMP containers

References are used as they are, so every container works: `Array`, `DenseAxisArray`,
`SparseAxisArray` and `Dict`. A slice such as `x[k, :]` works on the containers that support it
(`Array` and `DenseAxisArray`); for the others, list the elements:
`variables = [x[k, t] for t in T]`. Scalars and containers can be mixed.

## Pure master variables

A variable assigned to the master is a *pure master variable*. Its cost,
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
