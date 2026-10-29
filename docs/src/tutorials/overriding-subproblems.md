# Tutorial: overriding a subproblem

By default Vertigo builds a MathOptInterface model for every subproblem and solves it with the
optimiser of your JuMP model. Each of these subproblems is overridable by the user to allow ad-hoc
algorithms. This is particularly interesting when the subproblems have particular structures:
knapsack problems, shortest path with resource contraints problems, etc. When a subproblem is
overridden, Vertigo.jl builds no model for it, and the user's algorithm is responsible for generating
the columns.

This tutorial replaces the knapsack subproblems of the [GAP tutorial](gap.md) with a greedy
heuristic while this heuristic computes negative reduced cost columns and a dynamic program to
ensure optimality.

!!! warning

    Vertigo.jl stops executing once no reduced cost columns are returned by the subproblems. In the
    case of overridden subproblems, the user is responsible of ensuring his algorithm allows
    Vertigo.jl to prove optimaly if desired. The user is also responsible of ensuring that his
    algorithm generates columns that satisfy the subproblem's constraints.

## 1. Column generation

This tutorial reuses the flat model and the decomposition of [GAP tutorial](gap.md):

```julia
using JuMP, HiGHS, Vertigo

cost = [3.0 13.0 16.0 4.0 5.0 5.0 11.0 18.0;
        5.0 11.0 17.0 4.0 9.0 8.0 9.0 9.0]
weight = [6.0 5.0 3.0 1.0 2.0 2.0 2.0 2.0;
          3.0 2.0 2.0 2.0 1.0 4.0 6.0 6.0]
capacity = [13.0, 14.0]
K = 1:2; T = 1:8

model = Model(HiGHS.Optimizer)
set_silent(model)
@variable(model, x[k in K, t in T], Bin)
@constraint(model, assign[t in T], sum(x[k, t] for k in K) == 1)
@constraint(model, knapsack[k in K], sum(weight[k, t] * x[k, t] for t in T) <= capacity[k])
@objective(model, Min, sum(cost[k, t] * x[k, t] for k in K, t in T))

decomp, sp_map = dantzig_wolfe_decomposition(
    model,
    MasterProblemUserDescription(constraints = assign),
    [
        SubproblemUserDescription(k; variables = x[k, :], constraints = knapsack[k]) for
        k in K
    ],
)
```

At each column generation iteration, Vertigo.jl solves the master problem and gets its dual values.
For every subproblem, it then asks for the column with the lowest *reduced cost*:

TODO EQUATIONS

This is the default behavior of Vertigo.jl which solves the subproblems using the same solver as the
flat model. However, negative reduced cost columns obtained through quick-running heuristics often
improve performances. These algorithms can be implemented by overriding the subproblem's solving
algorithm.

## 2. Declaring the override

Give the subproblems the option `user_overridden = true` to override them:

```julia
decomp, sp_map = dantzig_wolfe_decomposition(
    model,
    MasterProblemUserDescription(constraints = assign),
    [
        SubproblemUserDescription(
            k;
            variables = x[k, :],
            constraints = knapsack[k],
            options = SubproblemOptions(user_overridden = true),
        ) for k in K
    ],
)
```

!!! note

    Note that it is possible to override only a subset of problems, with, for instance,
    `user_override = (k % 2 == 0)`

TODO: Detail the link between user override and JuMP variables.

<!-- Three things change compared with an ordinary subproblem: -->
<!---->
<!-- - its variables still exist, numbered `1..n`, and still appear in the coupling -->
<!--   constraints of the master. -->

## 3. Writing the pricer

The pricer stores the data it needs and the current reduced costs.

```julia
using JuMP, HiGHS, Vertigo
import MathOptInterface as MOI
const CG = Vertigo.ColGen

mutable struct KnapsackPricer
    weight::Vector{Int}
    capacity::Int
    sp_vars::Vector{MOI.VariableIndex}  # sp_vars[t]: Vertigo's variable for task t
    reduced_cost::Vector{Float64}       # current reduced cost of every task
end
```

### The variables of the subproblem: `sp_vars`

Your pricer thinks in tasks, `t = 1..8`. Vertigo thinks in
`MOI.VariableIndex`es: the variables of the subproblem, as it numbers them.
`sp_vars` is the dictionary between the two: `sp_vars[t]` is the
`MOI.VariableIndex` that stands for the JuMP variable `x[k, t]` inside
subproblem `k`. It is used in both directions:

- **Reduced costs in:** Vertigo identifies a variable by its
  `MOI.VariableIndex` when it writes a reduced cost. The pricer finds the
  matching task with `findfirst(==(change.variable), p.sp_vars)`.
- **Columns out:** a column is a list of `(variable index, value)` pairs, so the
  pricer turns the tasks it chose back into indices with `p.sp_vars[t]`.

You obtain the indices from the decomposition, with
`mapped_subproblem_var(decomp, x[k, t])`. It returns the subproblem and the
index of that variable, so `last(...)` keeps the index. For `x[1, 3]` and
`x[2, 3]` it returns, for example,

```julia
mapped_subproblem_var(decomp, x[1, 3])   # (PricingSubproblemId(1), MOI.VariableIndex(3))
mapped_subproblem_var(decomp, x[2, 3])   # (PricingSubproblemId(2), MOI.VariableIndex(3))
```

### What the indices are for an overridden subproblem

An overridden subproblem has no model, so its indices do not refer to a
variable of any MOI model. They are labels that Vertigo hands out itself: the
variables assigned to the subproblem are numbered `1, 2, ..., n`, `n` being
their number, taking the variable names in alphabetical order and, for each
name, its indices in sorted order. Here, subproblem `k` holds the 8 variables
`x[k, 1], ..., x[k, 8]`, hence `sp_vars == [VariableIndex(1), ..., VariableIndex(8)]`.

(For a subproblem that is not overridden, the indices are those of the
variables in the MOI model Vertigo builds for it.)

Three points follow from this:

- The indices of each subproblem are numbered from 1, so the same number
  appears in different subproblems. An index identifies a variable only
  together with its subproblem, which is why each subproblem gets its own
  pricer, with its own `sp_vars`.
- If a subproblem holds several variable names, they are numbered one after
  the other, whatever their declaration order. With `x[k, t]` and `y[k, t]`
  and three tasks, the `x` variables get indices 1 to 3 and the `y` variables
  4 to 6.
- Do not assume that `sp_vars[t]` equals `t`, even when it happens to. Always
  build `sp_vars` with `mapped_subproblem_var`. The reverse mapping,
  `mapped_original_var(decomp, sp_id, index)`, gives back the JuMP variable.

### Receiving the reduced costs

Vertigo writes each reduced cost with `MOI.modify`, as it would on a real
model. Implement it to store the value:

```julia
function MOI.modify(
    p::KnapsackPricer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    change::MOI.ScalarCoefficientChange{Float64}
)
    p.reduced_cost[findfirst(==(change.variable), p.sp_vars)] = change.new_coefficient
    return nothing
end
```

### Solving the subproblem

A standard dynamic programme over the tasks and the remaining capacity:

```julia
function solve_knapsack(p::KnapsackPricer)
    n, C = length(p.weight), p.capacity
    best = zeros(n + 1, C + 1)   # best[t+1, c+1]: tasks 1..t, capacity c
    for t in 1:n, c in 0:C
        best[t+1, c+1] = best[t, c+1]
        if p.weight[t] <= c
            best[t+1, c+1] = min(best[t+1, c+1], best[t, c+1-p.weight[t]] + p.reduced_cost[t])
        end
    end
    chosen, c = Int[], C
    for t in n:-1:1
        if best[t+1, c+1] != best[t, c+1]
            push!(chosen, t)
            c -= p.weight[t]
        end
    end
    return best[n+1, C+1], sort(chosen)
end
```

It returns the optimal value and the chosen tasks.

### Answering the pricing call

Two methods tell column generation how to use the pricer. The first says that
the same pricer is used in the exact stage:

```julia
CG.get_pricing_subprob_optimizer(::CG.ExactStage, sp::CG.PricingSubproblem{<:KnapsackPricer}) =
    CG.moi_pricing_sp(sp)
```

The second does the work:

```julia
function CG.optimize_pricing_problem!(
    ws::CG.ColGenWorkspace, sp_id, ::CG.PricingSubproblem, p::KnapsackPricer,
    mast_dual_sol, _stab_changes_mast_dual_sol
)
    decomp = ws.decomp
    ν = CG._get_convexity_dual(mast_dual_sol, convexity_lb_pairs(decomp), sp_id) +
        CG._get_convexity_dual(mast_dual_sol, convexity_ub_pairs(decomp), sp_id)
    value, chosen = solve_knapsack(p)
    column_reduced_cost = value + subproblem_fixed_cost(decomp, sp_id) - ν
    solution = Vertigo.Reformulation._SpSolution(
        sp_id, column_reduced_cost, [(p.sp_vars[t], 1.0) for t in chosen]
    )
    column = CG.PricingPrimalSolution(sp_id, solution, column_reduced_cost < -CG.RC_IMPROVING_TOL)
    return CG.PricingSolution(false, false, column_reduced_cost, value, [column])
end
```

Step by step:

- `ν` is the dual of the subproblem's convexity constraints. Subtracting it,
  and adding the subproblem's fixed cost, turns the knapsack value into the
  column's reduced cost.
- The column is the list of `(variable, value)` pairs with a non-zero value.
- A column is *improving* when its reduced cost is below `-CG.RC_IMPROVING_TOL`
  (for a minimisation problem). Only improving columns enter the master.
- `PricingSolution(is_infeasible, is_unbounded, primal_bound, dual_bound, columns)`
  has two bounds on the subproblem. `primal_bound` is the reduced cost of the
  best column found. `dual_bound` is a lower bound on the subproblem optimum,
  and drives the dual bound of the whole decomposition. With an exact method
  the two coincide, here at `value` (before the fixed cost and `ν`).

> **Note.** The pricing interface uses names that are not exported:
> `CG._get_convexity_dual`, `Vertigo.Reformulation._SpSolution` and
> `CG.RC_IMPROVING_TOL`. They may change without notice while the API is under
> development.

## 4. Installing the pricers and solving

Once the decomposition exists, create one pricer per machine and install it.
`mapped_subproblem_var(decomp, x[k, t])` returns the subproblem and the
variable index Vertigo uses for `x[k, t]`:

```julia
for k in K
    sp_vars = [last(mapped_subproblem_var(decomp, x[k, t])) for t in T]
    sp_models(decomp)[sp_map[k]] = KnapsackPricer(
        Int.(weight[k, :]), Int(capacity[k]), sp_vars, zeros(length(T))
    )
end

output = run_column_generation(ColGenWorkspace(decomp, ColGenConfig()))

output.status                # optimal
output.master_lp_obj         # 63.5
output.incumbent_dual_bound  # 63.5
```

The result is the same as with the subproblems solved by HiGHS in the
[GAP tutorial](gap.md).

Branch-and-price accepts the same decomposition. Build a fresh one, install the
pricers again, and run:

```julia
output = run_branch_and_price(BranchCutPriceWorkspace(decomp, BranchCutPriceConfig()))

output.incumbent.obj_value   # 65.0
output.root_lp_value         # 63.5
output.nodes_explored        # 5
```

The same incumbent is found in the same number of nodes.

## 5. Heuristic pricing

A pricer does not have to be exact. Returning a good column quickly is often
enough while column generation is far from convergence. Two rules keep the
results correct:

- Set `dual_bound` to a **valid lower bound** on the subproblem optimum, such
  as the linear relaxation of a knapsack. It is used to compute the dual bound
  of the decomposition. An invalid value gives an invalid bound.
- Report a column as improving only when its reduced cost really is negative.
  When no improving column is found, column generation stops. If the dual bound
  and the master value still differ then, Vertigo warns that optimality is not
  proven. It means the pricer stopped without a proof, not that the master is
  wrong.

## Where next

- [Decomposition rules](../user-api/decomposition.md) for the options of
  `subproblem(...)`.
- [Solving](../user-api/solving.md) for the column generation and
  branch-and-price configuration.
