# Decomposition Modeling Language

The DML (`src/DecompModelLanguage/`) turns a JuMP model and a description of the master and of the
subproblems into a `DWDecomposition`. It is the only part of Vertigo that reads JuMP objects;
everything after it works on MathOptInterface models and indices.

!!! note

    The DML and the internals are being reworked. The `@dantzig_wolfe` macro is not part of the
    build at the moment; it will be adapted to build the descriptions described below.

## From the descriptions to the reformulation

```text
+= the user's script ========================================================================+
|                                                                                            |
| model = Model(); @variable(model, ...); @constraint(model, ...)                            |
|       |                                                                                    |
|       v                                                                                    |
| describe the master and each subproblem, with their options      UserProblemDescription.jl |
|       |   master      = MasterProblemUserDescription(variables = ..., constraints = ...)   |
|       |   subproblems = [SubproblemUserDescription(id; variables = ..., ...), ...]         |
|       v                                                                                    |
| decomp, sp_map = dantzig_wolfe_decomposition(model, master, subproblems)                   |
|                                                                                            |
+====================================|=======================================================+
                                     |
                                     v
+= DantzigWolfe.jl: dantzig_wolfe_decomposition(model, master, subproblems) ================+
|                                                                                            |
|   +- Checker.jl -----------------------------------------------------------------------+   |
|   | _check_subproblem_ids, _check_ownerships                                           |   |
|   +------------------------------------------------------------------------------------+   |
|       |                                                                                    |
|       v                                                                                    |
|   +- Mappings.jl ----------------------------------------------------------------------+   |
|   | create the mappings between the flat model and the decomposed models               |   |
|   +------------------------------------------------------------------------------------+   |
|       |                                                                                    |
|       v                                                                                    |
|   +- Transition.jl --------------------------------------------------------------------+   |
|   | copy the objects into the JuMP master and subproblem models                        |   |
|   +------------------------------------------------------------------------------------+   |
|       |                                                                                    |
|       v                                                                                    |
|       fill the Decomposition builder                                                       |
|                                                                                            |
+====================================|=======================================================+
                                     |
                                     |  hand the data over to the Decomposition module
                                     v
                  Decomposition module returns a DWDecomposition object
```

The [Decomposition page](decomposition.md) follows the flow from the hand-over on.

## Basic structures

The user describes what each model contains with the JuMP objects themselves:

```julia
SubproblemUserDescription(k; variables = x[k, :], constraints = knapsack[k])
```

**User descriptions** (`UserProblemDescription.jl`) hold one model each:

| Field | `MasterProblemUserDescription` | `SubproblemUserDescription` |
|-------|-------------------------------|-----------------------------|
| `id` | none: the master has no id | the identifier of the subproblem; any value that can be sorted |
| `options` | a `MasterProblemOptions` | a `SubproblemOptions` |
| `owned_vars` | the `JuMP.VariableRef`s of the model that it owns | the same |
| `owned_constrs` | the `JuMP.ConstraintRef`s of the model that it owns | the same |

The keyword constructors take `variables` and `constraints` as a reference, a container of
references or a nested vector of them, and flatten them into `owned_vars` and `owned_constrs`.
A reference of the wrong kind is rejected there, by `_collect_varconstrs_refs!`.

**Options** (`UserProblemDescription.jl`) are plain structures with defaults, one value per model:

| Structure | Fields |
|-----------|--------|
| `MasterProblemOptions` | `silent = true` |
| `SubproblemOptions` | `silent = true`, `user_overridden = false` |

**Mappings** (`Mappings.jl`) `VariableMapping` and `ConstraintMapping` link each
object of the original JuMP model to its copy in the master or a subproblem
model while the new models are built.

Every structure of the DML:

| Structure | Defined in | Holds |
|-----------|------------|-------|
| `MasterProblemOptions`, `SubproblemOptions` | `UserProblemDescription.jl` | the options of one model |
| `MasterProblemUserDescription` | `UserProblemDescription.jl` | the options, the variables and the constraints of the master |
| `SubproblemUserDescription` | `UserProblemDescription.jl` | the id, the options, the variables and the constraints of one subproblem |
| `VariableMapping`, `ConstraintMapping` | `Mappings.jl` | the link between each original object and its copy |

The docstrings of these structures are in the [API Reference](../reference/dml.md).

## Steps

The chart above, in words.

1. **Describing** (`UserProblemDescription.jl`). The user builds one description for the master and one for each
   subproblem. Nothing is checked against the model yet, except that `variables` holds variable
   references and `constraints` constraint references.
2. **Checking** (`Checker.jl`). `dantzig_wolfe_decomposition` rejects an empty collection of
   subproblems and two subproblems with the same id. `_check_ownerships` then goes through the
   descriptions: every reference must belong to the decomposed model and be owned once; every
   variable and constraint of the model must be owned; and a subproblem constraint may use the
   variables of its own subproblem only. A failure is an `ArgumentError` naming the object. See
   [Errors](../user-api/errors.md).
3. **Building** (`Transition.jl`, `DantzigWolfe.jl`). The owned objects are taken in the
   order of the model, so the order the user wrote them in does not change the result.
   `_create_reformulated_jump_models` copies them into JuMP models; the options decide whether
   a model is built and whether it is silent. `_build_decomposition` then fills the Decomposition
   builder (`_fill_builder`) and calls `build(builder, master, sps)` with the MOI models. The
   Decomposition module builds the `DWDecomposition`, adds the convexity constraints to the
   master (`add_convexity_constraints!`) and stores the models; see the
   [master](master.md) and [subproblem](subproblem.md) pages.

## Files

| File | Role |
|------|------|
| `UserProblemDescription.jl` | the options structures and the user descriptions of the master and of a subproblem |
| `Checker.jl` | checks that the descriptions are correct before anything is built: distinct ids, one owner for each variable and constraint, no foreign variable in a subproblem constraint |
| `Mappings.jl` | the variable and constraint mappings that link each object of the flat model to its copy |
| `Transition.jl` | the functions that copy variables, constraints and the objective of the flat model into the new JuMP master and subproblem models |
| `DantzigWolfe.jl` | `dantzig_wolfe_decomposition`, the entry point: it runs the checks, then `_create_reformulated_jump_models` builds the master and subproblem models and `_build_decomposition` registers everything in the `DWDecomposition` |
| `macros/dantzig_wolfe.jl` | the `@dantzig_wolfe` macro and its parser, written for an earlier shape of the descriptions; not part of the build until it is adapted |

## Details worth knowing

- **Order.** Variables and constraints are copied in the order of the JuMP model, whatever the
  order of the lists given to the descriptions.
- **Overridden subproblems.** For `user_overridden = true`, no model is built and the
  variables get fresh `MOI.VariableIndex` values `1, 2, …`, in the order of the model. See
  `_subproblem_variable_indices` in `DantzigWolfe.jl`.
- **Names.** The copies keep the name of the original object, if it has one.
