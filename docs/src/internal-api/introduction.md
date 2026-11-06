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
4. **Branch and price** (`src/BranchCutPrice/`). When an integer solution is
   wanted, `run_branch_and_price` (`space.jl`) searches a tree; each node is
   evaluated by `BPEvaluator` (`evaluator.jl`), which runs column generation. Its
   `Branching` submodule (`src/BranchCutPrice/Branching/`) chooses the original
   variable to branch on (most fractional, strong branching with LP and
   column generation probes, or a user priority) from the master solution; `create_branching_children`
   (`branching.jl`) turns the choice into the two children.
   The module relies on three others, besides the decomposition it solves:
   - **Column generation** (`src/ColGen/`) solves the linear relaxation of
     every node. Its workspace also holds the active branching constraints and
     the incumbent.
   - **TreeSearch** (`src/TreeSearch/`) is the generic search: nodes, the
     search loop and the node selection strategies (depth first, best first,
     plunging, and others, including a user priority). It
     knows nothing of optimisation; `BranchCutPriceWorkspace` is its search
     space and `BPEvaluator` its node evaluator.
   - **MathOptState** (`src/MathOptState/`) records the differences between
     the model states of two nodes (variable bounds, local cuts), so that the
     master model is moved from one node to another without being rebuilt. A
     branching decision is a local cut added by such a difference.
5. **Return to user.** Column generation returns a `ColGenOutput`
   (`src/ColGen/context.jl`) and branch and price a `BPOutput`
   (`src/BranchCutPrice/bp_output.jl`).

```text
+-----------------------------------------+
|               (User API)                |
| DecompModelLanguage Module              |
| - Entry point: JuMP compact model       |
| and master and subproblem descriptions  |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+
|             (Internal API)              |
| Decomposition Module                    |
| - Construction of master problem,       |
| subproblems and mappings between user   |
| structures and internal structures      |
+-----------------------------------------+
                     |
                     v
+-----------------------------------------+      +----------------------------------------+
| Integer solution desired by user ?      |--no->| ColGen Module                          |--+
+-----------------------------------------+      +----------------------------------------+  |
                     |                                                                       |
                    yes                                                                      |
                     |                                                                       |
                     v                                                                       |
+-----------------------------------------+      +----------------------------------------+  |
| BranchCutPrice Module                   |      | TreeSearch Module                      |  |
|                                         |<---->| in:  node status, children             |  |
|                                         |      | out: the node to evaluate              |  |
| - Includes the Branching submodule      |      +----------------------------------------+  |
|                                         |                                                  |
|                                         |      +----------------------------------------+  |
|                                         |      | MathOptState Module                    |  |
|                                         |<---->| in:  node diffs                        |  |
|                                         |      | out: active cuts                       |  |
|                                         |      +----------------------------------------+  |
|                                         |                                                  |
|                                         |      +----------------------------------------+  |
|                                         |      | ColGen Module                          |  |
|                                         |<---->| in:  incumbent bound, branching cuts   |  |
|                                         |      | out: ColGenOutput (LP, bounds, IP)     |  |
+-----------------------------------------+      +----------------------------------------+  |
                     |                                                                       |
                     v                                                                       |
+-----------------------------------------+                                                  |
| Return to user                          |                                                  |
| - BPOutput (branch and price) or        |                                                  |
| ColGenOutput (column generation)        |<-------------------------------------------------+
+-----------------------------------------+
```


TODO: Why different objects ?
