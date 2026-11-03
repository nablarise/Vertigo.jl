# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    MasterProblemOptions(; silent = true)

Options of the master problem. With `silent = true`, the solver of the master model prints
nothing.
"""
Base.@kwdef struct MasterProblemOptions
    silent::Bool = true
end

"""
    SubproblemOptions(; silent = true, user_overridden = false)

Options of one subproblem.

- `silent`: with `true`, the solver of the subproblem model prints nothing.
- `user_overridden`: with `true`, the subproblem is solved by an algorithm of your own and no
  MOI model is built for it. Its constraints are not modelled, since your algorithm enforces
  them. Its variables still get a `MOI.VariableIndex`, available through
  `mapped_subproblem_var`, and `sp_models(decomp)[sp_id]` is `nothing` until you install
  your pricer there.
"""
Base.@kwdef struct SubproblemOptions
    silent::Bool = true
    user_overridden::Bool = false
end

"""
    MasterProblemUserDescription(variables = (), constraints = (), options = MasterProblemOptions())

What the master problem contains: the variables and the constraints of the JuMP model that
it owns, and its options. `variables` and `constraints` each take a reference, a container of
references or slices of containers such as `x[k, :]`, possibly nested in a vector.

The fields are `options`, `owned_vars` (a `Vector{JuMP.VariableRef}`) and `owned_constrs`
(a `Vector{JuMP.ConstraintRef}`).

Throws an `ArgumentError` if something that is not a variable reference is given as a
variable, or something that is not a constraint reference as a constraint.

# Example
```julia
MasterProblemUserDescription(
    variables = u, constraints = [assign, max_unassign],
    options = MasterProblemOptions(silent = false),
)
```
"""
struct MasterProblemUserDescription
    options::MasterProblemOptions
    owned_vars::Vector{JuMP.VariableRef}
    owned_constrs::Vector{JuMP.ConstraintRef}
end

function MasterProblemUserDescription(;
    variables=(), constraints=(), options=MasterProblemOptions()
)
    return MasterProblemUserDescription(
        options,
        _collect_varconstrs_refs(variables, JuMP.VariableRef, "variable"),
        _collect_varconstrs_refs(constraints, JuMP.ConstraintRef, "constraint"),
    )
end

"""
    SubproblemUserDescription(id; variables = (), constraints = (), options = SubproblemOptions())

What one subproblem contains: its identifier `id`, the variables and the constraints of the
JuMP model that it owns, and its options. `variables` and `constraints` are given as in
[`MasterProblemUserDescription`](@ref). The identifier can be any value that can be sorted
together with the identifiers of the other subproblems; it is the key of the `sp_id_map`
returned by [`dantzig_wolfe_decomposition`](@ref).

The fields are `id`, `options`, `owned_vars` and `owned_constrs`.

# Example
```julia
[
    SubproblemUserDescription(
        k;
        variables = x[k, :],
        constraints = knapsack[k],
        options = SubproblemOptions(user_overridden = (k == 1)),
    ) for k in K
]
```
"""
struct SubproblemUserDescription
    id::Any
    options::SubproblemOptions
    owned_vars::Vector{JuMP.VariableRef}
    owned_constrs::Vector{JuMP.ConstraintRef}
end

function SubproblemUserDescription(
    id; variables=(), constraints=(), options=SubproblemOptions()
)
    return SubproblemUserDescription(
        id,
        options,
        _collect_varconstrs_refs(variables, JuMP.VariableRef, "variable"),
        _collect_varconstrs_refs(constraints, JuMP.ConstraintRef, "constraint"),
    )
end

"""
    _collect_varconstrs_refs(container, R, what) -> Vector{R}

Flatten what the user gives as `variables` or `constraints` into a vector of references.
`R` is the type of reference expected (`JuMP.VariableRef` or `JuMP.ConstraintRef`) and `what`
its name in the error message (`"variable"` or `"constraint"`).

`container` can be a reference, or a container of references: an array, a JuMP container
(`DenseAxisArray`, `SparseAxisArray`), a dictionary (its values are taken) or a tuple. A
container can hold other containers, at any depth. The references keep the order in which they
are met, and are neither sorted nor made unique: that is done when the model is known.

Throws an `ArgumentError` if an element is neither a reference of type `R` nor a container,
for instance a constraint given as a variable.

It is used by the keyword constructors of [`MasterProblemUserDescription`](@ref) and
[`SubproblemUserDescription`](@ref), so that a description takes what JuMP gives: `x[k, :]`,
`assign` or `knapsack[k]`, without a call to `collect`.

# Example
```julia
_collect_varconstrs_refs(x[1, :], JuMP.VariableRef, "variable")               # the references of the slice
_collect_varconstrs_refs([assign, max_unassign], JuMP.ConstraintRef, "constraint")  # a container and one reference
_collect_varconstrs_refs(assign, JuMP.VariableRef, "variable")                # ArgumentError: a constraint
```
"""
function _collect_varconstrs_refs(container, ::Type{R}, what::String) where {R}
    return _collect_varconstrs_refs!(R[], container, what)
end

# Fills `refs` with the references held by `container`; see `_collect_varconstrs_refs`.
function _collect_varconstrs_refs!(refs::Vector{R}, container, what::String) where {R}
    if container isa R
        push!(refs, container)
    elseif container isa Union{AbstractArray,AbstractDict,Tuple}
        for element in (container isa AbstractDict ? values(container) : container)
            _collect_varconstrs_refs!(refs, element, what)
        end
    else
        throw(
            ArgumentError(
                "expected a $what or a container of them, got `$container` " *
                "of type $(typeof(container))",
            ),
        )
    end
    return refs
end
