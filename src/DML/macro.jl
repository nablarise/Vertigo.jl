# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# The arguments of `subproblem(...)`, passed as written to
# `dantzig_wolfe_subproblem`.
struct SubproblemAssignment
    args::Vector{Any}
end

struct MasterAssignment
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
- `subproblem(id; multiplicity = (lb, ub))` — and bound the number of its
  solutions used; see [`dantzig_wolfe_subproblem`](@ref)
- `master()` — assign to master problem

The arguments of `subproblem` may use the captured indices and the variables
of the calling scope.

# Example
```julia
@dantzig_wolfe model begin
    x[m, _] => subproblem(m)
    knapsack[m] => subproblem(m)
    coverage[_] => master()
end
```

Returns `(decomp::DWReformulation, sp_id_map::Dict)`. Solver parameters are
set on the resulting models; see [`dantzig_wolfe_decomposition`](@ref).
"""
macro dantzig_wolfe(model, block)
    patterns = parse_block(block)
    fn_name = gensym("generated_annotation")
    method_defs = generate_method_definitions(fn_name, patterns)

    # Escaped, so that the arguments of `subproblem` see the caller's
    # variables; the functions of this module are interpolated as values.
    return esc(quote
        $(method_defs...)
        $(dantzig_wolfe_decomposition)($model, $fn_name)
    end)
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
            pattern, assignment = line.args[2], line.args[3]
            push!(patterns, parse_pattern_rule(pattern, assignment))
        else
            error("Expected pattern => assignment, got: $line")
        end
    end

    isempty(patterns) && error(
        "No patterns specified in @dantzig_wolfe block"
    )
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
    if assignment.head == :call
        func_name = assignment.args[1]
        if func_name == :subproblem
            args = assignment.args[2:end]
            positional = filter(
                arg -> !Meta.isexpr(arg, (:parameters, :kw)), args
            )
            length(positional) == 1 || error(
                "subproblem() requires exactly one positional argument"
            )
            return SubproblemAssignment(args)
        elseif func_name == :master
            length(assignment.args) == 1 || error(
                "master() takes no arguments"
            )
            return MasterAssignment()
        else
            error("Unknown assignment function: $func_name")
        end
    else
        error(
            "Expected function call for assignment, got: $assignment"
        )
    end
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
    val_param = :(::$(Val){$(QuoteNode(pattern.name))})
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
    return :($(dantzig_wolfe_subproblem)($(a.args...)))
end

function generate_assignment_expr(::MasterAssignment)
    return :($(dantzig_wolfe_master)())
end
