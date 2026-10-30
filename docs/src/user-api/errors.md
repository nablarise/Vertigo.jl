# Errors

!!! warning

    This page is experimental and has not been reviewed.

`dantzig_wolfe_decomposition` checks the descriptions before building anything. Each message
names the offending object.

| Situation | Error | What to do |
|-----------|-------|------------|
| A variable or constraint is in no description | `ArgumentError`: *constraint `knapsack[1]` is not assigned to the master or to a subproblem* | list it in the master or in a subproblem |
| A variable or constraint is in two descriptions, or twice in one | `ArgumentError`: *`knapsack[1]` is assigned more than once, to the master and to subproblem 1…* | keep it in one description |
| Two subproblems have the same identifier | `ArgumentError`: *the subproblem 1 is described more than once…* | give each subproblem its own identifier |
| No subproblem is given | `ArgumentError`: *At least one subproblem is needed to decompose the model* | describe the subproblems |
| A reference belongs to another model | `ArgumentError`: *`x[1]` of the master does not belong to the model that is decomposed* | use the objects of the model you decompose |
| A constraint is given as a variable, or the reverse | `ArgumentError`: *expected a variable or a container of them, got…* | give `variables` variable references and `constraints` constraint references |
| A subproblem constraint uses a variable of another model | `ArgumentError`: *constraint `mixed` belongs to subproblem 1 but uses variable `x[2,1]` of subproblem 2…* | move the constraint to the master, or the variable to the same subproblem |
| A `user_overridden` subproblem has no pricer | error from `ColGenWorkspace(decomp, config)`: *has no pricer* | install a pricer in `sp_models(decomp)` |

## Notes

- A variable or constraint needs no name: it is found by its reference, so an anonymous
  one can be listed like any other.
- The foreign-variable check exists because a subproblem constraint is built
  over its own subproblem's variables only. A term on any other variable would
  be dropped and change the model. A constraint that links subproblems belongs
  in the master.
