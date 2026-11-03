# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    dantzig_wolfe_decomposition(model, master, subproblems; optimizer=nothing)

Decompose a JuMP `model` into a Dantzig-Wolfe reformulation. `master` is a
[`MasterProblemUserDescription`](@ref) and `subproblems` a collection of
[`SubproblemUserDescription`](@ref): together they say which model owns each variable and
constraint of `model`. Each variable and each constraint must be owned by exactly one of them.

Returns `(decomp::DWDecomposition, sp_id_map::Dict)` where `sp_id_map`
maps the identifiers of the subproblem descriptions to `PricingSubproblemId`.

# Example
```jldoctest
using JuMP, HiGHS, Vertigo

model = Model()
@variable(model, x[1:2, 1:3], Bin)
@constraint(model, assign[j in 1:3], sum(x[m,j] for m in 1:2) >= 1)
@constraint(model, cap[m in 1:2], sum(x[m,j] for j in 1:3) <= 2)
@objective(model, Min, sum(x))

decomp, sp_map = dantzig_wolfe_decomposition(
    model,
    MasterProblemUserDescription(constraints = assign),
    [SubproblemUserDescription(m; variables = x[m, :], constraints = cap[m]) for m in 1:2],
)

length(collect(subproblem_ids(decomp)))

# output

2
```

# User-overridden subproblems
A subproblem described with `SubproblemOptions(user_overridden = true)` gets no MOI model: it
is solved by an algorithm of your own. Its variables are still numbered `1..n`
and still appear in the coupling constraints of the master. Install a pricer
for each such subproblem before creating the `ColGenWorkspace`, otherwise
the workspace constructor raises an error.

Use `mapped_subproblem_var(decomp, original_var)` to find the variable a
pricer must use for an original variable, and `mapped_original_var` for the
reverse mapping.

```jldoctest
using JuMP, HiGHS, Vertigo

model = Model()
@variable(model, x[1:2, 1:3], Bin)
@constraint(model, assign[j in 1:3], sum(x[m,j] for m in 1:2) >= 1)
@constraint(model, cap[m in 1:2], sum(x[m,j] for j in 1:3) <= 2)
@objective(model, Min, sum(x))

decomp, sp_map = dantzig_wolfe_decomposition(
    model,
    MasterProblemUserDescription(constraints = assign),
    [
        SubproblemUserDescription(
            m;
            variables = x[m, :],
            constraints = cap[m],
            options = SubproblemOptions(user_overridden = true),
        ) for m in 1:2
    ],
)

# No pricer yet: install one with `sp_models(decomp)[sp_map[m]] = pricer`.
sp_models(decomp)[sp_map[1]] === nothing

# output

true
```
"""
function dantzig_wolfe_decomposition(
    model::JuMP.Model, master::MasterProblemUserDescription, subproblems; optimizer=nothing
)
    subproblems = collect(SubproblemUserDescription, subproblems)
    if isempty(subproblems)
        throw(ArgumentError("At least one subproblem is needed to decompose the model"))
    end

    _check_subproblem_ids(subproblems)
    _check_ownerships(model, master, subproblems)

    jump_models = _create_reformulated_jump_models(model, master, subproblems, optimizer)
    return _build_decomposition(model, master, subproblems, jump_models)
end

# Creates the JuMP models of the master and of the subproblems that are not user-overridden,
# with the copies of the variables, the constraints and the objective. Returns the models
# (indexed by the identifiers of the subproblem descriptions) and the two mappings from the
# original objects to their copies.
function _create_reformulated_jump_models(model, master, subproblems, optimizer)
    master_vars = _in_model_order(master.owned_vars)
    master_constrs = _in_model_order(master.owned_constrs)

    sp_vars = Dict(sp.id => _in_model_order(sp.owned_vars) for sp in subproblems)
    sp_constrs = Dict(sp.id => _in_model_order(sp.owned_constrs) for sp in subproblems)
    sp_options = Dict(sp.id => sp.options for sp in subproblems)

    master_model = JuMP.Model()
    JuMP.set_objective_sense(master_model, JuMP.objective_sense(model))

    var_mapping = VariableMapping()
    constr_mapping = ConstraintMapping()

    _copy_variables!(master_vars, master_model, var_mapping)

    sp_models = Dict(
        sp_id => JuMP.Model() for
        sp_id in keys(sp_vars) if !sp_options[sp_id].user_overridden
    )
    for (sp_id, sp_model) in sp_models
        _copy_variables!(sp_vars[sp_id], sp_model, var_mapping)
    end

    for (sp_id, sp_model) in sp_models
        _copy_constraints!(sp_constrs[sp_id], sp_model, var_mapping, constr_mapping)
    end

    _copy_constraints!(master_constrs, master_model, var_mapping, constr_mapping)

    for sp_model in values(sp_models)
        _copy_objective!(model, sp_model, var_mapping; is_master=false)
    end

    _copy_objective!(model, master_model, var_mapping; is_master=true)

    opt = optimizer
    if opt === nothing
        opt = _get_optimizer(model)
    end

    if opt !== nothing
        JuMP.set_optimizer(master_model, opt)
        if master.options.silent
            JuMP.set_silent(master_model)
        end
        for (sp_id, sp_model) in sp_models
            JuMP.set_optimizer(sp_model, opt)
            if sp_options[sp_id].silent
                JuMP.set_silent(sp_model)
            end
        end
    end

    return (; master_model, sp_models, var_mapping, constr_mapping)
end

# Registers in a Decomposition builder what the original `model` and the mappings say about
# the subproblems, the coupling constraints and the pure master variables. Returns the builder
# and the map from the identifiers of the subproblem descriptions to `PricingSubproblemId`.
function _fill_builder(model, master, subproblems, jump_models)
    (; var_mapping, constr_mapping) = jump_models

    master_vars = _in_model_order(master.owned_vars)
    master_constrs = _in_model_order(master.owned_constrs)

    sp_vars = Dict(sp.id => _in_model_order(sp.owned_vars) for sp in subproblems)
    user_overridden_sp_ids = Set(sp.id for sp in subproblems if sp.options.user_overridden)

    # Internal identifiers 1..n, given to the user identifiers in sorted order.
    sorted_ids = sort([sp.id for sp in subproblems])
    sp_id_map = Dict(id => PricingSubproblemId(i) for (i, id) in enumerate(sorted_ids))

    minimize = (JuMP.objective_sense(model) == MOI.MIN_SENSE)
    builder = DWDecompositionBuilder{JuMP.VariableRef}(; minimize=minimize)

    for id in sorted_ids
        add_subproblem!(builder, sp_id_map[id], 0.0, 1.0, 1.0)
    end

    # Collect original objective coefficients
    orig_obj = JuMP.objective_function(model)
    orig_costs = _extract_costs(orig_obj)

    # Add the subproblem variables
    sp_var_index = _subproblem_variable_indices(
        sp_vars, var_mapping, user_overridden_sp_ids
    )
    for (user_id, vars) in sp_vars
        sp_id = sp_id_map[user_id]
        for orig_var in vars
            _, moi_idx = sp_var_index[orig_var]
            cost = get(orig_costs, orig_var, 0.0)
            add_sp_variable!(builder, sp_id, moi_idx, cost)
            add_mapping!(builder, orig_var, sp_id, moi_idx)
        end
    end

    # Add the coupling constraints and coefficients
    _add_coupling_data!(builder, master_constrs, constr_mapping, sp_id_map, sp_var_index)

    # Add the pure master variables
    _add_pure_master_data!(
        builder, master_vars, master_constrs, constr_mapping, var_mapping, orig_costs
    )

    return (builder, sp_id_map)
end

# Builds the `DWDecomposition` of the original `model` from the JuMP models, and returns it
# with the map from the identifiers of the subproblem descriptions to `PricingSubproblemId`.
function _build_decomposition(model, master, subproblems, jump_models)
    builder, sp_id_map = _fill_builder(model, master, subproblems, jump_models)

    # A user-overridden subproblem has no model: the user installs a pricer.
    sp_backends = Dict{PricingSubproblemId,Any}(
        sp_id_map[sp.id] => (
            if haskey(jump_models.sp_models, sp.id)
                JuMP.backend(jump_models.sp_models[sp.id])
            else
                nothing
            end
        ) for sp in subproblems
    )

    decomp = build(builder, JuMP.backend(jump_models.master_model), sp_backends)
    return (decomp, sp_id_map)
end

# The variables or constraints in the order they were created in the JuMP model. The order
# of a description is the order its author wrote, which must not change the models.
function _in_model_order(refs)
    return sort(refs; by=ref -> JuMP.index(ref).value)
end

# original variable => (user subproblem id, variable index in that subproblem).
# A user-overridden subproblem has no model to hand out indexes, so it numbers its
# variables 1, 2, ... like a fresh MOI model would.
function _subproblem_variable_indices(sp_vars, var_mapping, user_overridden_sp_ids)
    sp_var_index = Dict{JuMP.VariableRef,Tuple{Any,MOI.VariableIndex}}()
    for (user_id, vars) in sp_vars
        next_index = 0
        for orig_var in vars
            moi_idx = if user_id in user_overridden_sp_ids
                MOI.VariableIndex(next_index += 1)
            else
                JuMP.index(var_mapping[orig_var])
            end
            sp_var_index[orig_var] = (user_id, moi_idx)
        end
    end
    return sp_var_index
end

function _get_optimizer(model::JuMP.Model)
    be = JuMP.backend(model)
    if be isa MOI.Utilities.CachingOptimizer
        opt = be.optimizer
        if opt !== nothing
            inner = _unwrap_optimizer(opt)
            if inner !== nothing
                return typeof(inner)
            end
        end
    end
    return nothing
end

_unwrap_optimizer(opt) = opt
function _unwrap_optimizer(opt::MOI.Bridges.LazyBridgeOptimizer)
    return opt.model
end

function _extract_costs(obj::JuMP.AffExpr)
    return Dict{JuMP.VariableRef,Float64}(var => coeff for (var, coeff) in obj.terms)
end

function _extract_costs(obj::JuMP.VariableRef)
    return Dict{JuMP.VariableRef,Float64}(obj => 1.0)
end

function _extract_costs(obj)
    return Dict{JuMP.VariableRef,Float64}()
end

function _add_coupling_data!(
    builder, master_constrs, constr_mapping, sp_id_map, sp_var_index
)
    for orig_constr in master_constrs
        reform_constr = constr_mapping[orig_constr]
        cstr_moi = JuMP.index(reform_constr)
        rhs = JuMP.normalized_rhs(reform_constr)
        add_coupling_constraint!(builder, cstr_moi, rhs)

        # Extract coupling coefficients from original
        # constraint (uses original variables)
        orig_constr_obj = JuMP.constraint_object(orig_constr)
        _add_coupling_from_func!(
            builder, orig_constr_obj.func, cstr_moi, sp_var_index, sp_id_map
        )
    end
end

function _add_coupling_from_func!(
    builder, func::JuMP.AffExpr, cstr_moi, sp_var_index, sp_id_map
)
    for (orig_var, coeff) in func.terms
        _add_coupling_coefficient!(
            builder, orig_var, coeff, cstr_moi, sp_var_index, sp_id_map
        )
    end
end

function _add_coupling_from_func!(
    builder, single_var::JuMP.VariableRef, cstr_moi, sp_var_index, sp_id_map
)
    return _add_coupling_coefficient!(
        builder, single_var, 1.0, cstr_moi, sp_var_index, sp_id_map
    )
end

# Master variables are not in `sp_var_index`: they carry no coefficient here.
function _add_coupling_coefficient!(
    builder, orig_var, coeff, cstr_moi, sp_var_index, sp_id_map
)
    if !(haskey(sp_var_index, orig_var))
        return nothing
    end
    user_id, moi_idx = sp_var_index[orig_var]
    return add_coupling_coefficient!(builder, sp_id_map[user_id], moi_idx, cstr_moi, coeff)
end

function _add_pure_master_data!(
    builder, master_vars, master_constrs, constr_mapping, var_mapping, orig_costs
)
    for orig_var in master_vars
        reform_var = var_mapping[orig_var]
        moi_idx = JuMP.index(reform_var)

        cost = get(orig_costs, orig_var, 0.0)
        lb = JuMP.has_lower_bound(reform_var) ? JuMP.lower_bound(reform_var) : 0.0
        ub = JuMP.has_upper_bound(reform_var) ? JuMP.upper_bound(reform_var) : Inf
        is_int = JuMP.is_integer(reform_var) || JuMP.is_binary(reform_var)

        if JuMP.is_binary(reform_var)
            lb = 0.0
            ub = 1.0
        end

        add_pure_master_variable!(builder, moi_idx, cost, lb, ub, is_int)

        # Find coupling constraints involving this variable
        for orig_constr in master_constrs
            orig_constr_obj = JuMP.constraint_object(orig_constr)
            coeff = _get_coeff_in_func(orig_constr_obj.func, orig_var)
            if !iszero(coeff)
                cstr_moi = JuMP.index(constr_mapping[orig_constr])
                add_pure_master_coupling!(builder, moi_idx, cstr_moi, coeff)
            end
        end
    end
end

function _get_coeff_in_func(func::JuMP.AffExpr, var)
    return get(func.terms, var, 0.0)
end

function _get_coeff_in_func(single_var::JuMP.VariableRef, var)
    return single_var == var ? 1.0 : 0.0
end

function _get_coeff_in_func(::Any, ::Any)
    return 0.0
end
