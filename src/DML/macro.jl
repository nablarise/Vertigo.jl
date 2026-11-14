# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

struct SubproblemAssignment
    id_expr::Any
    options::Vector{Expr}   # `key=value` keywords, forwarded as written
end

struct MasterAssignment
    options::Vector{Expr}
end

struct PatternRule
    name::Symbol
    indices::Vector{Symbol}
    assignment::Union{SubproblemAssignment,MasterAssignment}
end

"""
    @dantzig_wolfe model begin
        pattern => assignment
        ...
    end

Macro for declarative Dantzig-Wolfe decomposition specification.

# Pattern Syntax
- `name[index1, index2, ...]` — variable or constraint pattern
- `_` — wildcard for ignored indices
- `variable_name` — captured index used in assignment

# Assignment Syntax
- `subproblem(id)` — assign to subproblem with given ID
- `master()` — assign to master problem
- Both accept `silent=true|false` to silence (or not) the solver of the
  model they assign to. Unset, a model is silent.
- `subproblem` also accepts `user_overriden=true` for a subproblem solved by
  your own algorithm: no MOI model is built for it. Its constraints are not modeled
  (the algorithm enforces them), its variables get a `MOI.VariableIndex`
  (see `mapped_subproblem_var`), and `sp_models(decomp)[sp_id]` is `nothing`
  until you install your pricer there.
- Rules assigning to the same model must not contradict each other.

# Example
```julia
@dantzig_wolfe model begin
    x[m, _] => subproblem(m)
    knapsack[m] => subproblem(m, silent=false)
    coverage[_] => master()
end
```

# Containers
Instead of assigning each pattern, you can list what each model contains.
Options are then written once, on the container. A section takes one pattern
or a vector of them, and `variables`/`constraints` are only there for the
reader:
```julia
@dantzig_wolfe model begin
    master() => (constraints = coverage[_])
    subproblem(m, user_overriden=true) => (
        variables = [x[m, _]],
        constraints = knapsack[m]
    )
end
```
Both forms can be mixed; they are equivalent.

Returns `(decomp::DWReformulation, sp_id_map::Dict)`.
"""
macro dantzig_wolfe(model, block)
    patterns = parse_block(block)
    fn_name = gensym("generated_annotation")
    method_defs = generate_method_definitions(fn_name, patterns)

    return quote
        $(method_defs...)
        dantzig_wolfe_decomposition($(esc(model)), $fn_name)
    end
end

function parse_block(block::Expr)
    if block.head != :block
        error("Expected begin...end block")
    end

    patterns = PatternRule[]
    for line in block.args
        line isa LineNumberNode && continue
        if line isa Expr && line.head == :call &&
           length(line.args) == 3 && line.args[1] == :(=>)
            lhs, rhs = line.args[2], line.args[3]
            if lhs isa Expr && lhs.head == :call
                append!(patterns, parse_container(lhs, rhs))
            else
                push!(patterns, parse_pattern_rule(lhs, rhs))
            end
        else
            error(
                "Expected `pattern => assignment` or " *
                "`master()/subproblem(id) => (variables = ..., constraints = ...)`, " *
                "got: $line"
            )
        end
    end

    isempty(patterns) && error(
        "No patterns specified in @dantzig_wolfe block"
    )
    return patterns
end

# `master(opts) => (variables = ..., constraints = ...)`: every listed pattern
# gets the container's assignment. Each section holds one pattern or a vector
# of them; the sections only separate variables from constraints for the
# reader, both are optional and handled alike.
function parse_container(call::Expr, contents)
    assign = parse_assignment(call)
    sections = contents isa Expr && contents.head == :tuple ? contents.args : [contents]
    patterns = PatternRule[]
    for section in sections
        section isa Expr && section.head in (:kw, :(=)) &&
            section.args[1] in (:variables, :constraints) || error(
            "Expected `variables = ...` or `constraints = ...` in container, " *
            "got: $section"
        )
        listed = section.args[2]
        for pattern in (listed isa Expr && listed.head == :vect ? listed.args : [listed])
            name, indices = parse_pattern(pattern)
            push!(patterns, PatternRule(name, indices, assign))
        end
    end
    return patterns
end

function parse_pattern_rule(pattern, assignment::Expr)
    name, indices = parse_pattern(pattern)
    assign = parse_assignment(assignment)
    return PatternRule(name, indices, assign)
end

function parse_pattern(pattern)
    if pattern isa Symbol
        return pattern, Symbol[]
    elseif pattern isa Expr && pattern.head == :ref
        name = pattern.args[1]
        name isa Symbol || error(
            "Pattern name must be a symbol, got: $name"
        )
        index_symbols = Symbol[]
        for idx in pattern.args[2:end]
            idx isa Symbol || error(
                "Index must be a symbol or _, got: $idx"
            )
            push!(index_symbols, idx)
        end
        return name, index_symbols
    else
        error("Expected pattern[indices] or pattern, got: $pattern")
    end
end

function parse_assignment(assignment::Expr)
    assignment.head == :call || error(
        "Expected function call for assignment, got: $assignment"
    )
    func_name = assignment.args[1]
    positional, options = split_call_args(assignment.args[2:end])
    if func_name == :subproblem
        length(positional) == 1 || error(
            "subproblem() requires exactly one positional argument"
        )
        return SubproblemAssignment(positional[1], options)
    elseif func_name == :master
        isempty(positional) || error(
            "master() takes no positional arguments"
        )
        return MasterAssignment(options)
    else
        error("Unknown assignment function: $func_name")
    end
end

# Splits call arguments into positional ones and `key=value` options, which
# are forwarded as keywords (`dantzig_wolfe_subproblem` rejects unknown ones).
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

function generate_method_definitions(
    fn_name::Symbol, patterns::Vector{PatternRule}
)
    return [
        generate_method_definition(fn_name, p)
        for p in patterns
    ]
end

function generate_method_definition(
    fn_name::Symbol, pattern::PatternRule
)
    val_param = :(::Val{$(QuoteNode(pattern.name))})
    params = if isempty(pattern.indices)
        [val_param]
    else
        [val_param; pattern.indices...]
    end
    assignment_expr = generate_assignment_expr(pattern.assignment)
    return :(function $fn_name($(params...))
        $assignment_expr
    end)
end

function generate_assignment_expr(a::SubproblemAssignment)
    return Expr(
        :call, :dantzig_wolfe_subproblem,
        Expr(:parameters, a.options...), a.id_expr
    )
end

function generate_assignment_expr(a::MasterAssignment)
    return Expr(
        :call, :dantzig_wolfe_master, Expr(:parameters, a.options...)
    )
end
