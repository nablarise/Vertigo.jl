# Introduction

This section is for people who develop Vertigo. It describes the modules and how flow runs through
them.

The flow chart below follows a call from the user's model to the result. The first two blocks build
the decomposition; the rest solve it.

1. **Decomposition Modeling Language** (`src/DecompModelLanguage/`). The
   user describes the master and the subproblems with the JuMP variables and
   constraints each one owns. `dantzig_wolfe_decomposition` checks that every
   variable and constraint has exactly one owner and turns the result into a
   decomposition. The [DML page](dml.md) follows these steps
   file by file.
2. **Decomposition** (`src/Decomposition/`). `builder.jl` (`DWDecompositionBuilder`
   and `build`) assembles the master, the subproblems and the mappings between
   original and internal variables into a `DWDecomposition`
   (`dw_decomposition.jl`, `subproblem_data.jl`, `column_pool.jl`).
3. **Column generation** (`src/ColGen/`). When the user only wants the linear
   relaxation, `run_column_generation` (`context.jl`) solves it directly. The
   iteration itself is in `coluna.jl` and `dw_colgen_iteration.jl`.
4. **Branch and price** (`src/BranchCutPrice/`, `src/TreeSearch/`,
   `src/Branching/`). When an integer solution is wanted,
   `run_branch_and_price` (`space.jl`) searches a tree with `TreeSearch`; each
   node is evaluated by `BPEvaluator` (`evaluator.jl`), which runs column
   generation, and `Branching` creates the children.
5. **Return to user.** Column generation returns a `ColGenOutput`
   (`src/ColGen/context.jl`) and branch and price a `BPOutput`
   (`src/BranchCutPrice/bp_output.jl`).

```text
+---------------------------------------+
|              (User API)               |
| Decomposition Modeling Language       |
| - Entry point: JuMP compact model     |
| and master and subproblem descriptions|
+---------------------------------------+
                    |
                    v
+---------------------------------------+
|            (Internal API)             |
| Decomposition                         |
| - Construction of master problem,     |
| subproblems and mappings between user |
| structures and internal structures    |
+---------------------------------------+
                    |
                    v
+---------------------------------------+      +-------------------+
| Integer solution desired by user ?    |--no->| Column Generation |
+---------------------------------------+      +-------------------+
                    |                                    |
                   yes                                   |
                    |                                    |
                    v                                    |
+---------------------------------------+                |
| Branch and Price                      |                |
+---------------------------------------+                |
                    |                                    |
                    v                                    |
+---------------------------------------+                |
| Return to user                        |                |
| - BPOutput (branch and price) or      |                |
| ColGenOutput (column generation)      |<---------------+
+---------------------------------------+
```


TODO: Why different objects ?
