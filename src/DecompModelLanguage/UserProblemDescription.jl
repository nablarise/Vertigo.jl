# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

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
    MasterProblemUserDescription(options, owned_vars, owned_constrs)

What a `master(key=value, ...) => (variables = ..., constraints = ...)` line
says about the master, kept as unevaluated syntax. `options` are the
`key=value` keywords (`silent`). `owned_vars` and `owned_constrs` are the
slice descriptions listed under `variables` and `constraints`.
"""
struct MasterProblemUserDescription
    options::Vector{Expr}
    owned_vars::Vector{JuMPSliceDescription}
    owned_constrs::Vector{JuMPSliceDescription}
end

"""
    SubproblemUserDescription(id_expr, options, owned_vars, owned_constrs)

What a `subproblem(id, key=value, ...) => (variables = ..., constraints = ...)`
line says about one subproblem, kept as unevaluated syntax. `id_expr` is the id
expression as written; it may use an index symbol bound by a slice description, which
only gets a value when the generated method runs. `options` are the
`key=value` keywords (`silent`, `user_overriden`); their values may use the
index symbols too. `owned_vars` and `owned_constrs` are the slice descriptions listed
under `variables` and `constraints`.

The macro turns each owned slice description into an [`Ownership`](@ref), whose function
gives the id and options of the matching object.
"""
struct SubproblemUserDescription
    id_expr::Any
    options::Vector{Expr}   # `key=value` keywords, as written
    owned_vars::Vector{JuMPSliceDescription}
    owned_constrs::Vector{JuMPSliceDescription}
end

_owned_slices(d) = Iterators.flatten((d.owned_vars, d.owned_constrs))

"""
    parse_block(block) -> (master, subproblems)

Read the `begin ... end` block of a `@dantzig_wolfe` call, as unevaluated syntax. Each
line must have the form `master(...) => contents` or `subproblem(id, ...) => contents`;
it is read by [`parse_description`](@ref).

Returns the [`MasterProblemUserDescription`](@ref) and the vector of
[`SubproblemUserDescription`](@ref)s, in the order they are written. The block must
contain exactly one `master()` line.

Throws an error if `block` is not a `begin ... end` block, if a line has another form
(the form `x[k, _] => subproblem(k)` is not accepted), or if `master()` is missing or
written more than once.

# Example
```julia
master, subproblems = parse_block(:(
    begin
        master() => (constraints = assign[_])
        subproblem(k) => (variables = x[k, _], constraints = knapsack[k])
    end
))
```
"""
function parse_block(block::Expr)
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
            description = parse_description(line.args[2], line.args[3])
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
    parse_description(call, contents)

Read one line `call => contents` of a `@dantzig_wolfe` block. `call` is the expression
`master(options)` or `subproblem(id, options)`, and `contents` the expression after `=>`:
a tuple `(variables = ..., constraints = ...)`, or a single such section. Each section
holds one slice description or a vector of them, and both sections are optional. The
slice descriptions are read by `parse_slice`.

Returns a [`MasterProblemUserDescription`](@ref) or a [`SubproblemUserDescription`](@ref).
The options are `silent` for both, and `user_overriden` for a subproblem only.

Throws an error for a section that is neither `variables = ...` nor `constraints = ...`,
for an unknown option, for a `subproblem` without exactly one positional argument (its
id), for a `master` with a positional argument, and for any other function name.
"""
function parse_description(call::Expr, contents)
    sections = contents isa Expr && contents.head == :tuple ? contents.args : [contents]

    owned_vars = JuMPSliceDescription[]
    owned_constrs = JuMPSliceDescription[]

    for section in sections
        if !(
            section isa Expr &&
            section.head in (:kw, :(=)) &&
            section.args[1] in (:variables, :constraints)
        )
            error(
                "Expected `variables = ...` or `constraints = ...` in container, " *
                "got: $section",
            )
        end
        owned = section.args[1] == :variables ? owned_vars : owned_constrs
        listed = section.args[2]
        for slice in (listed isa Expr && listed.head == :vect ? listed.args : [listed])
            push!(owned, parse_slice(slice))
        end
    end

    func_name = call.args[1]
    positional, options = split_call_args(call.args[2:end])
    allowed = func_name == :master ? (:silent,) : (:silent, :user_overriden)

    for o in options
        if !(o.args[1] in allowed)
            error(
                "Unknown option `$(o.args[1])` for $func_name in @dantzig_wolfe, " *
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

function parse_slice(slice)
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

# Splits call arguments into positional ones and `key=value` options.
# Handles both `f(a, key=x)` and `f(a; key=x)`.
function split_call_args(args)
    positional = Any[]
    options = Expr[]
    for arg in args
        for a in (arg isa Expr && arg.head == :parameters ? arg.args : [arg])
            if a isa Expr && a.head in (:kw, :(=))
                push!(options, Expr(:kw, a.args[1], a.args[2]))
            else
                push!(positional, a)
            end
        end
    end
    return positional, options
end

# One `Ownership` per owned slice description. Its function takes the index values of an
# object and returns `(id, silent, user_overriden)`: the options may use the
# indices, so they are evaluated per object, like the id.
function generate_ownerships(master, subproblems)
    return [
        generate_ownership(p, d) for d in Iterators.flatten(((master,), subproblems)) for
        p in _owned_slices(d)
    ]
end

function generate_ownership(slice::JuMPSliceDescription, description)
    resolve = Expr(:->, Expr(:tuple, slice.indices...), generate_resolved_expr(description))
    return Expr(
        :call,
        GlobalRef(@__MODULE__, :Ownership),
        QuoteNode(slice.name),
        length(slice.indices),
        description isa MasterProblemUserDescription,
        resolve,
    )
end

# The value written for option `name`, or `nothing` when the line does not say.
function _option_expr(options, name)
    for o in options
        if o.args[1] == name
            return o.args[2]
        end
    end
    return nothing
end

function generate_resolved_expr(d::SubproblemUserDescription)
    silent = _option_expr(d.options, :silent)
    user_overriden = _option_expr(d.options, :user_overriden)
    return :((id=($(d.id_expr)), silent=($silent), user_overriden=($user_overriden)))
end

function generate_resolved_expr(d::MasterProblemUserDescription)
    silent = _option_expr(d.options, :silent)
    return :((id=nothing, silent=($silent), user_overriden=nothing))
end
