# Decomposition Modeling Language

The DML (`src/DecompModelLanguage/`) turns a JuMP model and a set of rules into a
`DWReformulation`. It is the only part of Vertigo that reads JuMP objects;
everything after it works on MathOptInterface models and indices.

!!! note

    The DML and the internals are being reworked.

## From the macro to the reformulation

`@dantzig_wolfe` works in two stages, and the difference between them is what makes the DML hard to
follow. The macro itself first expands to rewrite Julia syntax at compile time, a form of
metaprogramming. Then, at runtime, the code runs based on the user's JuMP model.


| Aspect | Stage 1: macro expansion | Stage 2: run time |
|--------|--------------------------|-------------------|
| When | the macro is expanded, before the script runs | the script reaches the `@dantzig_wolfe` line |
| What is known | the text of the block: names, index symbols, option expressions | the JuMP model, with its objects and their real indices |
| What is not known | the model, the index values; `model` is only an expression | --- |
| Result | the code `dantzig_wolfe_decomposition(model, [Ownership(...), ...])` | `(decomp, sp_id_map)` |

```text
 the user's script
 -----------------
   model = Model(); @variable(model, ...); @constraint(model, ...)
   decomp, sp_map = @dantzig_wolfe model begin ... end
                           |
                           |  the macro receives the expression `model`
                           |  and the unevaluated block
                           v
+= STAGE 1: macro expansion (syntax only, no JuMP model yet) ================+
|                                                                            |
|  @dantzig_wolfe(model, block)             src/macros/dantzig_wolfe.jl      |
|        |                                                                   |
|        v                                                                   |
|  parse_block(block)                       UserProblemDescription.jl        |
|        |   for each line:                                                  |
|        |     parse_description --> split_call_args  (the options)          |
|        |                       \-> parse_slice (a slice description each)  |
|        v                                                                   |
|  master      ::MasterProblemUserDescription                                |
|  subproblems ::Vector{SubproblemUserDescription}                           |
|        |                                                                   |
|        v                                                                   |
|  generate_ownerships                                                       |
|        |   generate_ownership        one call per JuMPSliceDescription     |
|        |   generate_resolved_expr    the id and options, as code           |
|        v                                                                   |
|  returns the code:                                                         |
|    dantzig_wolfe_decomposition(model, [Ownership(...), Ownership(...)])    |
+====================================|=======================================+
                                     |
                                     |  the script runs this code;
                                     |  `model` is now a JuMP.Model
                                     v
+= STAGE 2: run time (the JuMP model is known) ==============================+
|                                                                            |
|  dantzig_wolfe_decomposition(model, ownerships)    BridgeJuMPtoInternal.jl |
|        |                                                                   |
|        v                                                                   |
|  empty ownerships?     ArgumentError                                       |
|        |                                                                   |
|        v                                                                   |
|  _ownership_table      (name, number of indices) --> Ownership             |
|        |                                                                   |
|        v                                                                   |
|  _check_all_objects_named(model)          indexes.jl: _named_objects,      |
|        |                                  _holds, _index_tuples            |
|        v                                                                   |
|  partition                                                  partition.jl   |
|        |   for each named variable and constraint (name, idx):             |
|        |     _ownership(table, ...)       finds its Ownership              |
|        |     owner.resolve(idx...)      gives (id, silent, user_overriden) |
|        |   _master_variables, _partition_subproblem_variables,             |
|        |   _master_constraints, _partition_subproblem_constraints          |
|        v                                                                   |
|  _check_no_foreign_variables                                               |
|        |                                                                   |
|        v                                                                   |
|  copy the objects into JuMP master and subproblem models    models.jl      |
|        |   _requested_option reads user_overriden (no model if true)       |
|        |   _register_variables!, _register_constraints!,                   |
|        |   _register_objective!   (VariableMapping, ConstraintMapping)     |
|        |   _requested_option reads silent; the optimiser is set            |
|        v                                                                   |
|  feed the DWReformulationBuilder                  Reformulation (builder)  |
|        |   add_subproblem!, add_sp_variable!, add_mapping!,                |
|        |   _register_coupling_data!, _register_pure_master_data!, build    |
|        v                                                                   |
|  convexity constraints in the master, then set_models!                     |
+====================================|=======================================+
                                     |
                                     v
                  (decomp::DWReformulation, sp_id_map)
```

## Basic structures

A line of the `@dantzig_wolfe` block says what one model contains:

```julia
subproblem(k) => (variables = x[k, _], constraints = knapsack[k])
```

**User descriptions** (`UserProblemDescription.jl`) hold
one such line, kept as unevaluated syntax:

| Field | `MasterProblemUserDescription` | `SubproblemUserDescription` |
|-------|-------------------------------|-----------------------------|
| `id_expr` | none: the master has no id | the expression of the subproblem id; it may use an index symbol bound by a slice description |
| `options` | the `key=value` options written in `master(...)` | the `key=value` options written in `subproblem(...)`, forwarded as written |
| `owned_vars` | the slice descriptions listed under `variables` | the same |
| `owned_constrs` | the slice descriptions listed under `constraints` | the same |

**`JuMPSliceDescription`** is one entry of `owned_vars` or `owned_constrs`: a variable
or constraint of the JuMP model as the user writes it. The line above has one
variable slice description, `x[k, _]`, and one constraint slice description, `knapsack[k]`.

| Slice description | `name` | `indices` |
|---------|--------|-----------|
| `x[k, _]` | `:x` | `[:k, :_]` |
| `knapsack[k]` | `:knapsack` | `[:k]` |

- `name` is the name of the JuMP variable or constraint container, and
  `indices` the symbols written in the brackets, empty for a scalar object.
  A symbol is either `_`, a wildcard that matches every value of that
  position, or a name that is bound to the index value and can be used in
  `id_expr`.
- A `JuMPSliceDescription` holds no object of the model. It only describes which
  objects it covers; the objects are found later, when the generated function
  is called on each of them. It is syntax because the macro expands before any
  index has a value: `k` is not bound yet and `_` has no value.
- Only the name and the number of indices identify the objects a slice description
  covers: `x[k, _]` and `x[_, t]` cover the same objects. `dantzig_wolfe_decomposition` therefore
  rejects two ownerships with the same name and the same number of indices, so
  that each variable or constraint is owned by exactly one model.
- `owned_vars` and `owned_constrs` are both handled alike for now: a
  constraint listed under `variables` is accepted.
- The macro turns each owned slice description into an **`Ownership`**
  (`partition.jl`): the slice description's name and number of indices, whether the
  model is the master, and a function `resolve(idx...)`. Called with the index
  values of one object, `resolve` returns `(id, silent, user_overriden)`: the
  id of its subproblem (`nothing` for the master) and the options of its model,
  `nothing` for an option the line does not set. For `x[k, _]` above it is,
  roughly:

  ```julia
  Ownership(:x, 2, false, (k, _) -> (id = k, silent = nothing, user_overriden = nothing))
  ```

  The id and the options are evaluated per object because they may use the
  index symbols, as in `subproblem(k, silent = (k == 2))`.

`dantzig_wolfe_decomposition(model, ownerships)` looks an object up by name and
number of indices, so two ownerships with the same name and number of indices
are rejected.

**Mappings** (`models.jl`) `VariableMapping` and `ConstraintMapping` link each
object of the original JuMP model to its copy in the master or a subproblem
model while the new models are built.

Every structure of the DML, with the stage that creates it:

| Structure | Defined in | Created in | Holds |
|-----------|------------|------------|-------|
| `JuMPSliceDescription` | `UserProblemDescription.jl` | stage 1 | the name and the index symbols of one slice description |
| `MasterProblemUserDescription` | `UserProblemDescription.jl` | stage 1 | the options and the slice descriptions of the master line |
| `SubproblemUserDescription` | `UserProblemDescription.jl` | stage 1 | the id expression, the options and the slice descriptions of one subproblem line |
| `Ownership` | `partition.jl` | the code is generated in stage 1, the objects are built in stage 2 | a slice description's name and number of indices, `is_master`, and `resolve` |
| `VariableMapping`, `ConstraintMapping` | `models.jl` | stage 2 | the link between each original object and its copy |

The docstrings of these structures are in the [API Reference](../reference/dml.md).

In short, parsing turns text into user descriptions that own `JuMPSliceDescription`s,
compilation turns each slice description into an `Ownership`, and calling its function
on the model's objects gives the ids and options that decide the partition.

## Steps

The chart above, in words.

1. **Parsing** (stage 1). The macro reads each line of the block into a
   user description (`master()` or `subproblem(id)`, with its options) that owns
   a `JuMPSliceDescription` for every variable and constraint it lists. A line that does
   not have the form `master() => (...)` or `subproblem(id) => (...)` is
   rejected, and so is a missing or a second `master()` line.
2. **Slice descriptions become ownerships** (stage 1). For each owned slice description the macro
   generates the code that builds an `Ownership` when the script runs. Its
   function takes one argument per index. An `_` index is an
   unnamed argument, so it matches every value; a named index is bound and can
   be used in `subproblem(id)` and in the options. An option a model does not
   take (`user_overriden` on `master()`) is rejected here.
3. **Asking the rules** (stage 2). `dantzig_wolfe_decomposition(model, ownerships)`
   rejects an empty list of ownerships and two ownerships with the same name and number
   of indices. It then lists every named variable and constraint of the model and looks up its
   `Ownership` by name and number of indices. A missing one is reported as an
   `ArgumentError` naming the object. An error raised by an id or option
   expression of your own is not caught.
4. **Checks** (stage 2). Before building anything, the DML checks that every variable and
   constraint has a name, and that no subproblem constraint uses a variable of
   another model. See [Errors](../user-api/errors.md).
5. **Building** (stage 2). The partition is turned into MOI models and a
   `DWReformulation`; see `BridgeJuMPtoInternal.jl` and the
   [master](master.md) and [subproblem](subproblem.md) pages.

## Files

| File | Role |
|------|------|
| `UserProblemDescription.jl` | parses the block into user descriptions and `JuMPSliceDescription`s, builds the `Ownership`s |
| `src/macros/dantzig_wolfe.jl` | the `@dantzig_wolfe` macro itself; included by the DML module |
| `indexes.jl` | enumerates the real index tuples of any JuMP container (`Array`, `DenseAxisArray`, `SparseAxisArray`, `Dict`) |
| `partition.jl` | `Ownership` and its table; looks up the owner of each named object and groups the results; runs the unnamed-object and foreign-variable checks |
| `models.jl` | the variable and constraint mappings, and the functions that copy variables, constraints and the objective into the new JuMP models |
| `BridgeJuMPtoInternal.jl` | `dantzig_wolfe_decomposition`, the run-time entry point: it runs the checks and the partition, builds the master and subproblem models, and registers everything in the `DWReformulation` |

## Details worth knowing

- **Addressing by real indices.** `x[k in ["a", "b"], t in 3:5]` is addressed by
  `("a", 3)`, not by position. `indexes.jl` returns the index tuples as the
  user declared them for every JuMP container.
- **Scalars.** A scalar object keeps an index set of `Set{Tuple{}}`; the model
  builder dispatches on it.
- **Options left unset.** `silent` and `user_overriden` are `nothing` until a
  rule sets them. Two rules assigning to the same model must agree; unset means
  the default (silent, built as a MOI model).
- **Overridden subproblems.** For `user_overriden=true`, no model is built and
  the variables get fresh `MOI.VariableIndex` values `1, 2, …` (names in
  alphabetical order, then sorted indices). See `_subproblem_variable_indices`
  in `BridgeJuMPtoInternal.jl`.
