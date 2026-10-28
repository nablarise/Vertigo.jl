# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

_label(name, idx) = isempty(idx) ? string(name) : string(name, "[", join(idx, ", "), "]")

"""
    Ownership(name, nindices, is_master, resolve)

Says which model owns the objects of the container `name` with `nindices`
indices. `resolve(idx...)` takes the index values of one object and returns
`(id, silent, user_overridden)`: the id of its subproblem (`nothing` for the
master) and the options of its model, `nothing` for an option the rule does
not set. `@dantzig_wolfe` builds one per owned slice description.
"""
struct Ownership
    name::Symbol
    nindices::Int
    is_master::Bool
    resolve::Function
end

_owned_slices(d) = Iterators.flatten((d.owned_vars, d.owned_constrs))

"""
    generate_ownerships(master, subproblems) -> Vector{Expr}

Generate the code that builds the [`Ownership`](@ref)s of a `@dantzig_wolfe` block: one
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
generate_ownerships(master, subproblems)  # two expressions, `Ownership(:assign, ...)`
                                          # and `Ownership(:x, ...)`
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

Generate the call `Ownership(name, nindices, is_master, resolve)` for one slice description
`slice` of the master or subproblem `description`. `resolve` is an anonymous function with
one argument per index symbol of `slice`, whose body is made by
[`generate_resolved_expr`](@ref):

```julia
# subproblem(k) => (variables = x[k, _])
Ownership(:x, 2, false, (k, _) -> (id = k, silent = nothing, user_overridden = nothing))
```
"""
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

"""
    generate_resolved_expr(description) -> Expr

Generate the named tuple `(id = ..., silent = ..., user_overridden = ...)` returned by
the `resolve` function of an [`Ownership`](@ref). The expressions written in `description`
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

# At most one `Ownership` per container name and number of indices.
function _ownership_table(ownerships)
    table = Dict{Tuple{Symbol,Int},Ownership}()
    for o in ownerships
        key = (o.name, o.nindices)
        if haskey(table, key)
            throw(
                ArgumentError(
                    "`$(o.name)` with $(o.nindices) index(es) is matched by more than one " *
                    "rule; each variable or constraint needs exactly one",
                ),
            )
        end
        table[key] = o
    end
    return table
end

# The `Ownership` of `name[idx...]`.
function _ownership(table, kind::Symbol, name::Symbol, idx::Tuple)
    return get(table, (name, length(idx))) do
        return throw(
            ArgumentError(
                "no rule assigns $kind `$(_label(name, idx))` to the master or " *
                "to a subproblem",
            ),
        )
    end
end

# Every variable and constraint has to be reachable by name: the rules address
# objects by name, so an anonymous one would silently stay out of the
# decomposition.
function _check_all_objects_named(model)
    named_vars = Set(
        get_scalar_object(model, name, idx) for
        (name, idx) in _named_objects(model, JuMP.AbstractVariableRef)
    )
    named_constrs = Set(
        get_scalar_object(model, name, idx) for
        (name, idx) in _named_objects(model, JuMP.ConstraintRef)
    )
    for var in JuMP.all_variables(model)
        if !(var in named_vars)
            throw(
                ArgumentError(
                    "variable `$var` has no name the rules can address: declare it with a " *
                    "name, e.g. `@variable(model, y[1:3])`",
                ),
            )
        end
    end
    for constr in JuMP.all_constraints(model; include_variable_in_set_constraints=false)
        if !(constr in named_constrs)
            throw(
                ArgumentError(
                    "constraint `$constr` has no name the rules can address: declare it " *
                    "with a name, e.g. `@constraint(model, c[i in I], ...)`",
                ),
            )
        end
    end
    return nothing
end

_function_variables(f::JuMP.AffExpr) = keys(f.terms)
_function_variables(f::JuMP.VariableRef) = (f,)
_function_variables(_) = ()

# A subproblem constraint is built over its own subproblem's variables only;
# a term on any other variable would be dropped, changing the model.
function _check_no_foreign_variables(
    model, master_vars, sp_vars_partition, sp_constrs_partition
)
    owner = Dict{JuMP.VariableRef,Any}()
    for (name, indexes) in master_vars, idx in indexes
        owner[get_scalar_object(model, name, idx)] = :master
    end
    for (sp_id, by_name) in sp_vars_partition, (name, indexes) in by_name, idx in indexes
        owner[get_scalar_object(model, name, idx)] = (:subproblem, sp_id)
    end
    for (sp_id, by_name) in sp_constrs_partition, (name, indexes) in by_name, idx in indexes
        constr = get_scalar_object(model, name, idx)
        for var in _function_variables(JuMP.constraint_object(constr).func)
            if owner[var] == (:subproblem, sp_id)
                continue
            end
            where_ = owner[var] == :master ? "the master" : "subproblem $(owner[var][2])"
            throw(
                ArgumentError(
                    "constraint `$(_label(name, idx))` belongs to subproblem $sp_id but " *
                    "uses variable `$var` of $where_; a subproblem constraint can only " *
                    "use the variables of its own subproblem",
                ),
            )
        end
    end
    return nothing
end

# Scalars keep a `Set{Tuple{}}` of indices: the model builder dispatches on it.
function _push_index!(by_name::Dict{Symbol,Any}, name, idx)
    return push!(get!(() -> idx == () ? Set{Tuple{}}() : Set{Tuple}(), by_name, name), idx)
end

# What the rules assign `R`s (variables or constraints) to: `name => indexes`
# for the master, and `sp_id => (name => indexes)` for the subproblems.
function _master_objects(model, table, ::Type{R}, kind::Symbol) where {R}
    master = Dict{Symbol,Any}()
    for (name, idx) in _named_objects(model, R)
        _ownership(table, kind, name, idx).is_master && _push_index!(master, name, idx)
    end
    return master
end

function _subproblem_objects(model, table, ::Type{R}, kind::Symbol) where {R}
    partition = Dict{Any,Dict{Symbol,Any}}()
    for (name, idx) in _named_objects(model, R)
        owner = _ownership(table, kind, name, idx)
        if owner.is_master
            continue
        end
        id = owner.resolve(idx...).id
        _push_index!(get!(Dict{Symbol,Any}, partition, id), name, idx)
    end
    return partition
end

function _master_variables(model, table)
    return _master_objects(model, table, JuMP.AbstractVariableRef, :variable)
end
function _partition_subproblem_variables(model, table)
    return _subproblem_objects(model, table, JuMP.AbstractVariableRef, :variable)
end
function _master_constraints(model, table)
    return _master_objects(model, table, JuMP.ConstraintRef, :constraint)
end
function _partition_subproblem_constraints(model, table)
    return _subproblem_objects(model, table, JuMP.ConstraintRef, :constraint)
end
