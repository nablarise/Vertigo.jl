# Reformulation

The Reformulation module (`src/Reformulation/`) is the solver's own model of a
decomposed problem. It holds what column generation needs: the subproblems, the
coefficients that couple them to the master, the pure master variables, and the
mapping between the variables of the user's model and those of the
subproblems. It uses MathOptInterface only: it knows nothing about JuMP or about
the descriptions of the user. The [DML](dml.md) depends on it, and not the other
way round.

## From the builder to the algorithms

The DML hands the decomposition over piece by piece. Column generation, branching
and branch and price then ask questions of the result.

```text
   dantzig_wolfe_decomposition (DML), or any code that knows a decomposition
                           |
                           |  the pieces of the decomposition, one call each,
                           |  with MathOptInterface indices
                           v
+= BUILD ==================================================================================+
|                                                                                          |
|  DWReformulationBuilder{X}(; minimize)                                        builder.jl |
|        v                                                                                 |
|  add_subproblem!(sp_id, fixed_cost, conv_lb, conv_ub)                                    |
|  add_sp_variable!(sp_id, sp_var, cost)                                                   |
|  add_mapping!(orig_var, sp_id, sp_var)                                                   |
|  add_coupling_constraint!(cstr, rhs)                                                     |
|  add_coupling_coefficient!(sp_id, sp_var, cstr, coeff)                                   |
|  add_pure_master_variable!(y, cost, lb, ub, is_integer)                                  |
|  add_pure_master_coupling!(y, cstr, coeff)                                               |
|        v                                                                                 |
|  build(builder)                                                               builder.jl |
|        |   per subproblem: the coupling entries, sorted, in one flat array               |
|        |   per pure master variable: its data and coefficients                           |
|        |   both directions of the mapping between original and subproblem variables      |
|        v                                                                                 |
|  DWReformulation, without MOI models yet                             dw_reformulation.jl |
+====================================|=====================================================+
                                     |
                                     |  the caller adds the convexity constraints to the master
                                     |  and calls set_models!(decomp, master, sps, conv_ub, conv_lb)
                                     v
+= USE ====================================================================================+
|                                                                                          |
|  asking the decomposition                                            dw_reformulation.jl |
|        |   original_cost, coupling_coefficients                                          |
|        |   mapped_original_var, mapped_subproblem_var                                    |
|        |   compute_*: column costs, reduced costs, dual bound               interface.jl |
|        v                                                                                 |
|  recording the columns found                                              column_pool.jl |
|        |   record_column!, has_column (no duplicate column)                              |
|        v                                                                                 |
|  project_to_original: master values to original variables            dw_reformulation.jl |
+====================================|=====================================================+
                                     |
                                     v
                 column generation, branching, branch and price
```

## What it contains

| Element | Type | File |
|---------|------|------|
| The decomposition | `DWReformulation{X,M}`: subproblems, pure master variables, mapping, coupling constraints, sense, and the MOI models once set | `dw_reformulation.jl` |
| One subproblem | `SubproblemData`: variables, original costs, coupling coefficients, fixed cost, convexity bounds | `subproblem_data.jl` |
| One pure master variable | `PureMasterVariableData`: cost, bounds, integrality, coupling coefficients | `subproblem_data.jl` |
| Original and subproblem variables | `OneToOneMapping{X}`: both directions, where `X` is the type of the user's variable | `subproblem_data.jl` |
| Identifiers | `PricingSubproblemId`, `TaggedCI` (a constraint index that can be stored and sorted), `CouplingEntry` and `CouplingCoefficients` (the coefficients of a variable, as a slice of one flat array) | `moi_types.jl` |
| A subproblem solution | `_SpSolution`: sorted entries, and a fingerprint to find duplicates | `subproblem_solution.jl` |
| The columns found | `ColumnPool`, with one `ColumnRecord` per column | `column_pool.jl` |
| Robust cuts | `AbstractCutSeparator`, `SeparatedCut`, `CustomCutSeparator` | `cut_separation.jl` |

## Files

| File | Role |
|------|------|
| `interface.jl` | the abstract types, and the computations written against them: column cost, coupling coefficients, reduced costs, dual bound contribution |
| `moi_types.jl`, `moi_utils.jl` | the identifier types, and the functions that read the primal and dual solution of an MOI model |
| `subproblem_data.jl` | the data of a subproblem and of a pure master variable, and the variable mapping |
| `dw_reformulation.jl` | `DWReformulation`, its query functions, `set_models!` and `project_to_original` |
| `subproblem_solution.jl` | the subproblem solution type |
| `column_pool.jl` | the pool of columns, indexed by column variable, subproblem and fingerprint |
| `builder.jl` | `DWReformulationBuilder` and `build` |
| `cut_separation.jl` | the interface of a robust cut separator |
