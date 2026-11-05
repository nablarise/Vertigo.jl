# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    _check_subproblem_ids(subproblems)

Check that no two subproblem descriptions have the same id.

Throws an `ArgumentError` naming the id otherwise.
"""
function _check_subproblem_ids(subproblems)
    ids = Set{Any}()
    for description in subproblems
        if description.id in ids
            throw(
                ArgumentError(
                    "the subproblem $(description.id) is described more than once; " *
                    "each subproblem needs one description",
                ),
            )
        end
        push!(ids, description.id)
    end
    return nothing
end

"""
    _check_multiplicities(subproblems)

Check that the multiplicities of every subproblem are consistent: `0 <= lower_multiplicity <=
upper_multiplicity` and `upper_multiplicity >= 1` (a subproblem that can never be used has no
reason to be described).

Throws an `ArgumentError` naming the subproblem otherwise.
"""
function _check_multiplicities(subproblems)
    for description in subproblems
        lb = description.options.lower_multiplicity
        ub = description.options.upper_multiplicity
        if lb < 0 || ub < lb || ub < 1
            throw(
                ArgumentError(
                    "the multiplicities of the subproblem $(description.id) are " *
                    "($lb, $ub); they must satisfy 0 <= lower_multiplicity <= " *
                    "upper_multiplicity and upper_multiplicity >= 1",
                ),
            )
        end
    end
    return nothing
end

# The functions of a linear model: an affine expression, or a single variable (a bound).
const _LINEAR_FUNCTIONS = (JuMP.AffExpr, JuMP.VariableRef)

"""
    _check_linear(model)

Check that the objective and every constraint of `model` are linear: a decomposition of the
model assumes it. A constraint of another kind (quadratic, nonlinear, a vector function) or a
nonlinear objective would only fail later, while the master and the subproblem models are
built.

Throws an `ArgumentError` naming the objective, or the first constraint of the offending kind.

TODO: JuMP should answer this (a query on the model, such as whether it is a linear or mixed
integer linear program), not Vertigo. Replace this function when JuMP provides one.
"""
function _check_linear(model)
    objective_type = JuMP.objective_function_type(model)
    if !(objective_type in _LINEAR_FUNCTIONS)
        throw(
            ArgumentError(
                "the objective of the model is a $(objective_type); a decomposed model " *
                "must be linear",
            ),
        )
    end
    for (function_type, set_type) in JuMP.list_of_constraint_types(model)
        if function_type in _LINEAR_FUNCTIONS
            continue
        end
        constr = first(JuMP.all_constraints(model, function_type, set_type))
        throw(
            ArgumentError(
                "constraint `$(_label(constr))` is a $(function_type) in $(set_type); a " *
                "decomposed model must be linear",
            ),
        )
    end
    return nothing
end

"""
    _check_ownerships(model, master, subproblems)

Check that the descriptions split the variables and the constraints of `model` between the
master and the subproblems:

- every reference of every description belongs to `model`;
- every variable and every constraint of `model` is owned by exactly one description (a
  reference listed twice, by two descriptions or twice by one, is rejected, and so is one
  that no description lists);
- a subproblem constraint uses only the variables of its own subproblem. A term on any other
  variable would be dropped when the subproblem model is built, changing the constraint.

Throws an `ArgumentError` naming the offending object.
"""
function _check_ownerships(model, master, subproblems)
    descriptions = Iterators.flatten(((master,), subproblems))
    var_owner = Dict{JuMP.VariableRef,Any}()
    constr_owner = Dict{JuMP.ConstraintRef,Any}()
    for description in descriptions
        for var in description.owned_vars
            _assign!(var_owner, model, var, description)
        end
        for constr in description.owned_constrs
            _assign!(constr_owner, model, constr, description)
        end
    end

    for var in JuMP.all_variables(model)
        if !haskey(var_owner, var)
            throw(
                ArgumentError(
                    "variable `$(_label(var))` is not assigned to the master or to a " *
                    "subproblem",
                ),
            )
        end
    end
    for constr in JuMP.all_constraints(model; include_variable_in_set_constraints=false)
        if !haskey(constr_owner, constr)
            throw(
                ArgumentError(
                    "constraint `$(_label(constr))` is not assigned to the master or to " *
                    "a subproblem",
                ),
            )
        end
    end

    for description in subproblems, constr in description.owned_constrs
        for var in _function_variables(JuMP.constraint_object(constr).func)
            if var_owner[var] === description
                continue
            end
            throw(
                ArgumentError(
                    "constraint `$(_label(constr))` belongs to " *
                    "$(_owner_label(description)) but uses variable `$(_label(var))` of " *
                    "$(_owner_label(var_owner[var])); a subproblem constraint can only " *
                    "use the variables of its own subproblem",
                ),
            )
        end
    end
    return nothing
end

function _label(ref)
    name = JuMP.name(ref)
    return isempty(name) ? string(ref) : name
end

_owner_label(::MasterProblemUserDescription) = "the master"
_owner_label(description::SubproblemUserDescription) = "subproblem $(description.id)"

function _assign!(owner, model, ref, description)
    if JuMP.owner_model(ref) !== model
        throw(
            ArgumentError(
                "`$(_label(ref))` of $(_owner_label(description)) does not belong to " *
                "the model that is decomposed",
            ),
        )
    end
    if haskey(owner, ref)
        throw(
            ArgumentError(
                "`$(_label(ref))` is assigned more than once, to " *
                "$(_owner_label(owner[ref])) and to $(_owner_label(description)); " *
                "each variable or constraint needs exactly one owner",
            ),
        )
    end
    owner[ref] = description
    return nothing
end

_function_variables(f::JuMP.AffExpr) = keys(f.terms)
_function_variables(f::JuMP.VariableRef) = (f,)
_function_variables(_) = ()
