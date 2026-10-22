# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

struct SubproblemAssignment
    id_expr::Any
    silent_expr::Any
end

struct MasterAssignment
    silent_expr::Any
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
  model they assign to. Unset, a model is silent. Rules assigning to the
  same model must not contradict each other.

# Example
```julia
@dantzig_wolfe model begin
    x[m, _] => subproblem(m)
    knapsack[m] => subproblem(m, silent=false)
    coverage[_] => master()
end
```

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
    assignment.head == :call || error(
        "Expected function call for assignment, got: $assignment"
    )
    func_name = assignment.args[1]
    positional, silent_expr = split_call_args(assignment.args[2:end])
    if func_name == :subproblem
        length(positional) == 1 || error(
            "subproblem() requires exactly one positional argument"
        )
        return SubproblemAssignment(positional[1], silent_expr)
    elseif func_name == :master
        isempty(positional) || error(
            "master() takes no positional arguments"
        )
        return MasterAssignment(silent_expr)
    else
        error("Unknown assignment function: $func_name")
    end
end

# Splits call arguments into positional ones and the `silent` value.
# Handles both `f(a, silent=x)` and `f(a; silent=x)`.
function split_call_args(args)
    positional = Any[]
    silent_expr = :nothing
    for arg in args
        kws = if arg isa Expr && arg.head == :parameters
            arg.args
        else
            [arg]
        end
        for kw in kws
            if kw isa Expr && kw.head in (:kw, :(=)) && kw.args[1] == :silent
                silent_expr = kw.args[2]
            elseif kw isa Expr && kw.head in (:kw, :(=))
                error("Unknown keyword: $(kw.args[1])")
            else
                push!(positional, kw)
            end
        end
    end
    return positional, silent_expr
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
    return :(dantzig_wolfe_subproblem(
        $(a.id_expr); silent=$(a.silent_expr)
    ))
end

function generate_assignment_expr(a::MasterAssignment)
    return :(dantzig_wolfe_master(; silent=$(a.silent_expr)))
end
