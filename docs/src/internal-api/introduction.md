# Introduction

This section is for people who develop Vertigo. It describes the modules and how flow runs through
them.

The flow chart below follows a call from the user's model to the result. The first two blocks build
the decomposition; the rest solve it.

1. **Decomposition Modeling Language** (`src/DecompModelLanguage/`). `src/macros/dantzig_wolfe.jl` defines
   `@dantzig_wolfe`, `UserProblemDescription.jl` parses its block, `partition.jl` decides which part of
   the JuMP model belongs to the master or to a subproblem, and `BridgeJuMPtoInternal.jl`
   (`dantzig_wolfe_decomposition`) turns that partition into a decomposition.
2. **Reformulation** (`src/Reformulation/`). `builder.jl` (`DWReformulationBuilder`
   and `build`) assembles the master, the subproblems and the mappings between
   original and internal variables into a `DWReformulation`
   (`dw_reformulation.jl`, `subproblem_data.jl`, `column_pool.jl`).
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
| and @dantzig_wolfe macro              |
+---------------------------------------+
                    |
                    v
+---------------------------------------+
|            (Internal API)             |
| Reformulation                         |
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
