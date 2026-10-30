# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# The macro is not part of the build for now: the user API takes JuMP references
# (see UserProblemDescription.jl), and this code was written for the descriptions
# that held slice descriptions and expressions. It is kept here to be adapted: the
# macro will have to turn each slice description into references and build the
# descriptions of the user API.

"""
    JuMPSliceDescription(name, indices)

A variable or constraint of the JuMP model as written in a `@dantzig_wolfe`
block, such as `x[k, _]`. `name` is the name of the JuMP container (`:x`) and
`indices` the symbols written in the brackets (`[:k, :_]`, empty for a scalar
object). An index is either `_`, which matches every value, or a name bound to
the index value, which `subproblem(id)` can use.

It is syntax, not a JuMP object: the macro expands before any index has a
value. Only `name` and the number of indices select the objects it covers.
"""
struct JuMPSliceDescription
    name::Symbol
    indices::Vector{Symbol}
end

"""
    parse_decomposition_description(block) -> (master, subproblems)

Read the `begin ... end` block of a `@dantzig_wolfe` call, as unevaluated syntax. Each
line must have the form `master(...) => contents` or `subproblem(id, ...) => contents`;
each line is then read by [`parse_problem_description`](@ref).

Returns the [`MasterProblemUserDescription`](@ref) and the vector of
[`SubproblemUserDescription`](@ref)s, in the order they are written. The block must
contain exactly one `master()` line.

Throws an error if `block` is not a `begin ... end` block, if a line is ill-formed,
or if `master()` is missing or written more than once.

# Example
```julia
master, subproblems = parse_decomposition_description(:(
    begin
        master() => (constraints = assign[_])
        subproblem(k) => (variables = x[k, _], constraints = knapsack[k])
    end
))
```
"""
function parse_decomposition_description(block::Expr)
    if block.head != :block
        error("Expected begin...end block")
    end

    masters = MasterProblemUserDescription[]
    subproblems = SubproblemUserDescription[]

    for line in block.args
        if line isa LineNumberNode
            continue
        end

        # Detect `f() => ...` structure
        # line.args is a tuple of the form (:(=>), f(), ...).
        if (
            line isa Expr &&
            line.head == :call &&
            length(line.args) == 3 &&
            line.args[1] == :(=>) &&
            line.args[2] isa Expr &&
            line.args[2].head == :call
        )
            description = parse_problem_description(line.args[2], line.args[3])
            push!(
                description isa MasterProblemUserDescription ? masters : subproblems,
                description,
            )
        else
            error(
                "Expected `master()/subproblem(id) => " *
                "(variables = ..., constraints = ...)`, got: $line. " *
                "A slice description is listed in the model it belongs to: " *
                "`subproblem(k) => (variables = x[k, _])`, not " *
                "`x[k, _] => subproblem(k)`.",
            )
        end
    end

    if length(masters) != 1
        error(
            "@dantzig_wolfe needs exactly one master() line, got $(length(masters)); " *
            "list all the variables and constraints of the master in one master() line",
        )
    end

    master = only(masters)
    return master, subproblems
end

"""
    parse_problem_description(call, contents)

Read one line `call => contents` of a `@dantzig_wolfe` block. `call` is the expression
`master(options)` or `subproblem(id, options)`, and `contents` the expression after `=>`,
written in one of three ways:
- `(variables = ...)`: only variables;
- `(constraints = ...)`: only constraints;
- `(variables = ..., constraints = ...)`: both.

Each assignment holds one slice description (`x[k, _]`) or a vector of them
(`[x[k, _], y]`). The slice descriptions are read by [`parse_slice_description`](@ref).

Returns a [`MasterProblemUserDescription`](@ref) or a [`SubproblemUserDescription`](@ref).
The options are `silent` for both, and `user_overridden` for a subproblem only.

Throws an error for an assignment that is neither `variables = ...` nor `constraints = ...`,
for an unknown option, for a `subproblem` without exactly one positional argument (its
id), for a `master` with a positional argument, and for any other function name.
"""
function parse_problem_description(call::Expr, contents)
    # The contents may be a single assignment or a tuple of them. Single assignments are not
    # considered a tuple.
    assignments = contents isa Expr && contents.head == :tuple ? contents.args : [contents]

    owned_vars = JuMPSliceDescription[]
    owned_constrs = JuMPSliceDescription[]

    for assignment in assignments
        if !(
            assignment isa Expr &&
            assignment.head in (:kw, :(=)) &&
            assignment.args[1] in (:variables, :constraints)
        )
            error(
                "Expected `variables = ...` or `constraints = ...` in container, " *
                "got: $assignment",
            )
        end

        owned = assignment.args[1] == :variables ? owned_vars : owned_constrs
        listed = assignment.args[2]

        for slice in (listed isa Expr && listed.head == :vect ? listed.args : [listed])
            push!(owned, parse_slice_description(slice))
        end
    end

    func_name = call.args[1]
    # `key = value` in a call is parsed as `:kw`; every other argument is positional.
    is_option(arg) = arg isa Expr && arg.head == :kw
    options = Dict{Symbol,Any}(
        arg.args[1] => arg.args[2] for arg in call.args[2:end] if is_option(arg)
    )
    positional = filter(!is_option, call.args[2:end])
    allowed = func_name == :master ? (:silent,) : (:silent, :user_overridden)

    for name in keys(options)
        if !(name in allowed)
            error(
                "Unknown option `$name` for $func_name in @dantzig_wolfe, " *
                "expected one of: $(join(allowed, ", "))",
            )
        end
    end

    if func_name == :subproblem
        if !(length(positional) == 1)
            error("subproblem() requires exactly one positional argument")
        end

        return SubproblemUserDescription(positional[1], options, owned_vars, owned_constrs)
    elseif func_name == :master
        if !(isempty(positional))
            error("master() takes no positional arguments")
        end

        return MasterProblemUserDescription(options, owned_vars, owned_constrs)
    else
        error("Unknown model in @dantzig_wolfe: $func_name, expected master or subproblem")
    end
end

"""
    parse_slice_description(slice) -> JuMPSliceDescription

Read one slice description of a `@dantzig_wolfe` block, as unevaluated syntax. `slice`
is either a symbol naming a scalar variable or constraint (`z`), or an indexed name
`name[i1, i2, ...]` whose indices are symbols: `_`, or a name bound to the index value.

Returns a [`JuMPSliceDescription`](@ref) with the name and the index symbols. Nothing is
checked against a JuMP model, which does not exist yet when the macro expands.

Throws an error if `slice` is neither of the two forms, if the name of an indexed slice
description is not a symbol, or if an index is not a symbol (`x[1, _]` and `x[k + 1, _]`
are rejected).

# Example
```julia
parse_slice_description(:(x[k, _]))  # JuMPSliceDescription(:x, [:k, :_])
parse_slice_description(:z)          # JuMPSliceDescription(:z, Symbol[])
```
"""
function parse_slice_description(slice)
    if slice isa Symbol
        return JuMPSliceDescription(slice, Symbol[])
    elseif slice isa Expr && slice.head == :ref
        name = slice.args[1]
        if !(name isa Symbol)
            error("Name of a slice description must be a symbol, got: $name")
        end

        index_symbols = Symbol[]
        for idx in slice.args[2:end]
            if !(idx isa Symbol)
                error("Index must be a symbol or _, got: $idx")
            end
            push!(index_symbols, idx)
        end

        return JuMPSliceDescription(name, index_symbols)
    else
        error("Expected name[indices] or name, got: $slice")
    end
end

_owned_slices(d) = Iterators.flatten((d.owned_vars, d.owned_constrs))

"""
    generate_ownerships(master, subproblems) -> Vector{Expr}

Generate the code that builds the [`VarConstrOwnership`](@ref)s of a `@dantzig_wolfe` block: one
expression per slice description of the master and of each subproblem, in that order.
The expressions are not evaluated here. The macro puts them in the call to
`dantzig_wolfe_decomposition`, so they run when the script reaches the block, when the
JuMP model and the values of the user's own variables exist.

# Example
```julia
master, subproblems = parse_decomposition_description(:(
    begin
        master() => (constraints = assign[_])
        subproblem(k) => (variables = x[k, _])
    end
))
generate_ownerships(master, subproblems)  # two expressions, `VarConstrOwnership(:assign, ...)`
                                          # and `VarConstrOwnership(:x, ...)`
```
"""
function generate_ownerships(master, subproblems)
    return [
        generate_ownership(p, d) for d in Iterators.flatten(((master,), subproblems)) for
        p in _owned_slices(d)
    ]
end

"""
    generate_ownership(slice, description) -> Expr

Generate the call `VarConstrOwnership(name, nindices, is_master, resolve)` for one slice description
`slice` of the master or subproblem `description`. `resolve` is an anonymous function with
one argument per index symbol of `slice`, whose body is made by
[`generate_resolved_expr`](@ref):

```julia
# subproblem(k) => (variables = x[k, _])
VarConstrOwnership(:x, 2, false, (k, _) -> (id = k, silent = nothing, user_overridden = nothing))
```
"""
function generate_ownership(slice::JuMPSliceDescription, description)
    resolve = Expr(:->, Expr(:tuple, slice.indices...), generate_resolved_expr(description))
    return Expr(
        :call,
        GlobalRef(@__MODULE__, :VarConstrOwnership),
        QuoteNode(slice.name),
        length(slice.indices),
        description isa MasterProblemUserDescription,
        resolve,
    )
end

"""
    generate_resolved_expr(description) -> Expr

Generate the named tuple `(id = ..., silent = ..., user_overridden = ...)` returned by
the `resolve` function of an [`VarConstrOwnership`](@ref). The expressions written in `description`
are pasted in unevaluated: the id expression of a subproblem and the values of its
options, which may use the index symbols that `resolve` takes as arguments. An option the
line does not set is `nothing`, and so is the id of the master, which has none.
"""
function generate_resolved_expr(d::SubproblemUserDescription)
    silent = get(d.options, :silent, nothing)
    user_overridden = get(d.options, :user_overridden, nothing)
    return :((id=($(d.id_expr)), silent=($silent), user_overridden=($user_overridden)))
end

function generate_resolved_expr(d::MasterProblemUserDescription)
    silent = get(d.options, :silent, nothing)
    return :((id=nothing, silent=($silent), user_overridden=nothing))
end

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
- `subproblem` also accepts `user_overridden=true` for a subproblem solved by
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
    subproblem(m, user_overridden=true) => (
        variables = [x[m, _]],
        constraints = knapsack[m]
    )
end
```

Returns `(decomp::DWDecomposition, sp_id_map::Dict)`.
"""
macro dantzig_wolfe(model, block)
    master, subproblems = parse_decomposition_description(block)
    ownerships = generate_ownerships(master, subproblems)
    return quote
        dantzig_wolfe_decomposition($(esc(model)), $(esc(Expr(:vect, ownerships...))))
    end
end
