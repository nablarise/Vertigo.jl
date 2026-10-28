# Errors

!!! warning

    This page is experimental and has not been reviewed.

`@dantzig_wolfe` checks the model before building anything. Each message names
the offending object.

| Situation | Error | What to do |
|-----------|-------|------------|
| A variable or constraint matches no rule | `ArgumentError`: *no rule assigns constraint `knapsack[1]` to the master or to a subproblem* | add a rule covering it |
| `master()` is missing or written more than once | *@dantzig_wolfe needs exactly one master() line, got 0…* | write one `master()` line listing everything the master owns |
| Two rules match the same name and index count | `ArgumentError`: *`x` with 2 index(es) is matched by more than one rule…* | keep one rule |
| The block lists no variable or constraint | `ArgumentError`: *No slice descriptions specified in @dantzig_wolfe block* | list what each model owns |
| An object has no name | `ArgumentError`: *variable `…` has no name the rules can address* | declare it with a name: `@variable(model, y[1:3])` |
| A subproblem constraint uses a variable of another model | `ArgumentError`: *constraint `mixed` belongs to subproblem 1 but uses variable `x[2,1]` of subproblem 2…* | move the constraint to the master, or the variable to the same subproblem |
| A line is not `master()` or `subproblem(id)` followed by `=> (variables = ..., constraints = ...)` | `ErrorException` while parsing | fix the line |
| A `user_overriden` subproblem has no pricer | error from `ColGenWorkspace(decomp, config)`: *has no pricer* | install a pricer in `sp_models(decomp)` |

## Notes

- Unnamed objects are rejected because the rules work by name: an anonymous
  constraint would silently stay out of the decomposition.
- The foreign-variable check exists because a subproblem constraint is built
  over its own subproblem's variables only. A term on any other variable would
  be dropped and change the model. A constraint that links subproblems belongs
  in the master.
- An error raised by your own id or option expression (`subproblem(id)`,
  `silent = ...`) is not caught; only a missing rule is reported as above.
