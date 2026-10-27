# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    @dantzig_wolfe model begin
        master() => (variables = ..., constraints = ...)
        subproblem(id) => (variables = ..., constraints = ...)
        ...
    end

Macro for declarative Dantzig-Wolfe decomposition specification. Each line says
what one model contains: the variables and constraints it owns. Every named
variable and constraint of `model` must be owned by exactly one line.

# Models
- `master()` — the master problem
- `subproblem(id)` — the subproblem with given ID; `id` may use an index
  symbol bound by a slice description
- Both accept `silent=true|false` to silence (or not) the solver of the
  model. Unset, a model is silent.
- `subproblem` also accepts `user_overriden=true` for a subproblem solved by
  your own algorithm: no MOI model is built for it. Its constraints are not modeled
  (the algorithm enforces them), its variables get a `MOI.VariableIndex`
  (see `mapped_subproblem_var`), and `sp_models(decomp)[sp_id]` is `nothing`
  until you install your pricer there.
- Lines for the same model must not contradict each other.

# Owned variables and constraints
`variables` and `constraints` each take one slice description or a vector of them, and
both are optional:
- `name[index1, index2, ...]` — a variable or constraint container
- `name` — a scalar variable or constraint
- `_` — wildcard that matches every value of an index
- `k` — any other symbol is bound to the index value and can be used in `id`

# Example
```julia
@dantzig_wolfe model begin
    master() => (constraints = coverage[_])
    subproblem(m, user_overriden=true) => (
        variables = [x[m, _]],
        constraints = knapsack[m]
    )
end
```

Returns `(decomp::DWReformulation, sp_id_map::Dict)`.
"""
macro dantzig_wolfe(model, block)
    master, subproblems = parse_block(block)
    check_validity(master, subproblems)
    ownerships = generate_ownerships(master, subproblems)
    return quote
        dantzig_wolfe_decomposition($(esc(model)), $(esc(Expr(:vect, ownerships...))))
    end
end
