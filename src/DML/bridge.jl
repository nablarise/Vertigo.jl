# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    dantzig_wolfe_decomposition(model, dw_annotation; optimizer=nothing)

Decompose a JuMP `model` into a Dantzig-Wolfe reformulation using
the annotation function `dw_annotation`.

Returns `(decomp::DWReformulation, sp_id_map::Dict)` where `sp_id_map`
maps user-provided subproblem identifiers to `PricingSubproblemId`.

# Example
```jldoctest
using JuMP, HiGHS, Vertigo

model = Model()
@variable(model, x[1:2, 1:3], Bin)
@constraint(model, assign[j in 1:3], sum(x[m,j] for m in 1:2) >= 1)
@constraint(model, cap[m in 1:2], sum(x[m,j] for j in 1:3) <= 2)
@objective(model, Min, sum(x))

decomp, sp_map = @dantzig_wolfe model begin
    x[m, _] => subproblem(m)
    assign[_] => master()
    cap[m] => subproblem(m)
end

length(collect(subproblem_ids(decomp)))

# output

2
```

# User-overridden subproblems
A subproblem declared with `user_overriden=true` gets no MOI model: it is
solved by an algorithm of your own. Its variables are still numbered `1..n`
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

decomp, sp_map = @dantzig_wolfe model begin
    master() => (constraints = assign[_])
    subproblem(m, user_overriden=true) => (
        variables = x[m, _],
        constraints = cap[m]
    )
end

# No pricer yet: install one with `sp_models(decomp)[sp_map[m]] = pricer`.
sp_models(decomp)[sp_map[1]] === nothing

# output

true
```
"""
function dantzig_wolfe_decomposition(
    model::JuMP.Model, dw_annotation;
    optimizer=nothing
)
    master_vars = _master_variables(model, dw_annotation)
    sp_vars_partition = _partition_subproblem_variables(
        model, dw_annotation
    )
    master_constrs = _master_constraints(model, dw_annotation)
    sp_constrs_partition = _partition_subproblem_constraints(
        model, dw_annotation
    )

    # Create master JuMP model
    master_model = JuMP.Model()
    JuMP.set_objective_sense(
        master_model, JuMP.objective_sense(model)
    )

    var_mapping = VariableMapping()
    constr_mapping = ConstraintMapping()

    # Register master variables
    master_var_infos = Dict(
        var_name => Dict(
            idx => _original_var_info(model, var_name, idx)
            for idx in indexes
        )
        for (var_name, indexes) in master_vars
    )
    _register_variables!(
        master_model, var_mapping, model, master_var_infos
    )

    # Create subproblem JuMP models, except for user-overridden subproblems,
    # which are solved by the user's own algorithm and have no model.
    user_sp_ids = collect(keys(sp_vars_partition))
    sp_objects(sp_id) = Iterators.flatten((
        get(sp_vars_partition, sp_id, ()),
        get(sp_constrs_partition, sp_id, ())
    ))
    user_overriden_sp_ids = Set(
        sp_id for sp_id in user_sp_ids
        if _requested_option(
            sp_objects(sp_id), dw_annotation, :user_overriden, false,
            "subproblem $sp_id"
        )
    )
    sp_jump_models = Dict(
        sp_id => JuMP.Model()
        for sp_id in user_sp_ids if !(sp_id in user_overriden_sp_ids)
    )

    # Register subproblem variables
    sp_var_infos = Dict(
        sp_id => Dict(
            var_name => Dict(
                idx => _original_var_info(model, var_name, idx)
                for idx in indexes
            )
            for (var_name, indexes) in sp_vars_partition[sp_id]
        )
        for sp_id in keys(sp_jump_models)
    )

    for (sp_id, var_infos_by_names) in sp_var_infos
        _register_variables!(
            sp_jump_models[sp_id], var_mapping,
            model, var_infos_by_names
        )
    end

    # Register subproblem constraints
    for (sp_id, constr_by_names) in sp_constrs_partition
        sp_id in user_overriden_sp_ids && continue
        _register_constraints!(
            sp_jump_models[sp_id], constr_mapping,
            model, constr_by_names, var_mapping
        )
    end

    # Register master (coupling) constraints
    _register_constraints!(
        master_model, constr_mapping,
        model, master_constrs, var_mapping
    )

    # Register objectives
    for sp_model in values(sp_jump_models)
        _register_objective!(
            sp_model, model, var_mapping; is_master=false
        )
    end
    _register_objective!(
        master_model, model, var_mapping; is_master=true
    )

    # Resolve optimizer: explicit kwarg > original model's optimizer
    opt = optimizer
    if opt === nothing
        opt = _get_optimizer(model)
    end

    if opt !== nothing
        JuMP.set_optimizer(master_model, opt)
        _requested_option(
            Iterators.flatten((master_vars, master_constrs)),
            dw_annotation, :silent, true, "the master"
        ) && JuMP.set_silent(master_model)
        for (sp_id, sp_model) in sp_jump_models
            JuMP.set_optimizer(sp_model, opt)
            _requested_option(
                sp_objects(sp_id), dw_annotation, :silent, true,
                "subproblem $sp_id"
            ) && JuMP.set_silent(sp_model)
        end
    end

    # Deterministic sp_id mapping
    sorted_ids = sort(collect(user_sp_ids))
    sp_id_map = Dict(
        id => PricingSubproblemId(i)
        for (i, id) in enumerate(sorted_ids)
    )

    # Build DWReformulation via builder
    sense = JuMP.objective_sense(model)
    minimize = (sense == MOI.MIN_SENSE)
    builder = DWReformulationBuilder{JuMP.VariableRef}(
        minimize=minimize
    )

    for id in sorted_ids
        add_subproblem!(builder, sp_id_map[id], 0.0, 1.0, 1.0)
    end

    # Collect original objective coefficients
    orig_obj = JuMP.objective_function(model)
    orig_costs = _extract_costs(orig_obj)

    # Register subproblem variables and coupling coefficients
    sp_var_index = _subproblem_variable_indices(
        model, sp_vars_partition, var_mapping, user_overriden_sp_ids
    )
    for (user_id, var_by_names) in sp_vars_partition
        sp_id = sp_id_map[user_id]
        for (var_name, indexes) in var_by_names
            for idx in sort(collect(indexes))
                orig_var = get_scalar_object(model, var_name, idx)
                _, moi_idx = sp_var_index[orig_var]
                cost = get(orig_costs, orig_var, 0.0)
                add_sp_variable!(builder, sp_id, moi_idx, cost)
                add_mapping!(builder, orig_var, sp_id, moi_idx)
            end
        end
    end

    # Register coupling constraints and coefficients
    _register_coupling_data!(
        builder, model, master_model, master_constrs,
        constr_mapping, sp_id_map, sp_var_index
    )

    # Register pure master variables
    _register_pure_master_data!(
        builder, model, master_model, master_vars,
        master_constrs, constr_mapping, var_mapping,
        orig_costs
    )

    decomp = build(builder)

    # Add convexity constraints to master
    conv_ub_map = Dict{PricingSubproblemId,TaggedCI}()
    conv_lb_map = Dict{PricingSubproblemId,TaggedCI}()

    for id in sorted_ids
        sp_id = sp_id_map[id]
        conv_lb_ref = JuMP.@constraint(master_model, 0 >= 1)
        conv_ub_ref = JuMP.@constraint(master_model, 0 <= 1)
        conv_lb_map[sp_id] = TaggedCI(JuMP.index(conv_lb_ref))
        conv_ub_map[sp_id] = TaggedCI(JuMP.index(conv_ub_ref))
    end

    # Wire MOI backends
    # A user-overridden subproblem has no backend: the user installs a pricer.
    sp_backends = Dict{PricingSubproblemId,Any}(
        sp_id_map[id] => (
            haskey(sp_jump_models, id) ?
            JuMP.backend(sp_jump_models[id]) : nothing
        )
        for id in user_sp_ids
    )
    set_models!(
        decomp, JuMP.backend(master_model),
        sp_backends, conv_ub_map, conv_lb_map
    )

    return (decomp, sp_id_map)
end

# Value of annotation option `option` for the model gathering `objects`
# (name => indexes pairs). Rules that don't say are ignored; rules that do
# must agree, and `default` applies when none does.
function _requested_option(objects, dw_annotation, option, default, target)
    values = Set{Bool}()
    for (name, indexes) in objects, idx in indexes
        value = getproperty(dw_annotation(Val(name), idx...), option)
        value === nothing || push!(values, value)
    end
    length(values) > 1 && error(
        "Conflicting `$option` settings for $target: " *
        "all rules assigning to it must agree."
    )
    return isempty(values) ? default : only(values)
end

# original variable => (user subproblem id, variable index in that subproblem).
# A user-overridden subproblem has no model to hand out indexes, so it numbers its
# variables 1, 2, ... like a fresh MOI model would.
function _subproblem_variable_indices(
    model, sp_vars_partition, var_mapping, user_overriden_sp_ids
)
    sp_var_index = Dict{JuMP.VariableRef,Tuple{Any,MOI.VariableIndex}}()
    for (user_id, var_by_names) in sp_vars_partition
        next_index = 0
        for var_name in sort(collect(keys(var_by_names)))
            for idx in sort(collect(var_by_names[var_name]))
                orig_var = get_scalar_object(model, var_name, idx)
                moi_idx = if user_id in user_overriden_sp_ids
                    MOI.VariableIndex(next_index += 1)
                else
                    JuMP.index(var_mapping[orig_var])
                end
                sp_var_index[orig_var] = (user_id, moi_idx)
            end
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
            inner !== nothing && return typeof(inner)
        end
    end
    return nothing
end

_unwrap_optimizer(opt) = opt
function _unwrap_optimizer(
    opt::MOI.Bridges.LazyBridgeOptimizer
)
    return opt.model
end

function _extract_costs(obj::JuMP.AffExpr)
    return Dict{JuMP.VariableRef,Float64}(
        var => coeff for (var, coeff) in obj.terms
    )
end

function _extract_costs(obj::JuMP.VariableRef)
    return Dict{JuMP.VariableRef,Float64}(obj => 1.0)
end

function _extract_costs(obj)
    return Dict{JuMP.VariableRef,Float64}()
end

function _register_coupling_data!(
    builder, model, master_model, master_constrs,
    constr_mapping, sp_id_map, sp_var_index
)
    for (constr_name, constr_indexes) in master_constrs
        for idx in sort(collect(constr_indexes))
            orig_constr = get_scalar_object(
                model, constr_name, idx
            )
            reform_constr = constr_mapping[orig_constr]
            cstr_moi = JuMP.index(reform_constr)
            rhs = JuMP.normalized_rhs(reform_constr)
            add_coupling_constraint!(builder, cstr_moi, rhs)

            # Extract coupling coefficients from original
            # constraint (uses original variables)
            orig_constr_obj = JuMP.constraint_object(orig_constr)
            _add_coupling_from_func!(
                builder, orig_constr_obj.func, cstr_moi,
                sp_var_index, sp_id_map
            )
        end
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
    builder, single_var::JuMP.VariableRef, cstr_moi,
    sp_var_index, sp_id_map
)
    _add_coupling_coefficient!(
        builder, single_var, 1.0, cstr_moi, sp_var_index, sp_id_map
    )
end

# Master variables are not in `sp_var_index`: they carry no coefficient here.
function _add_coupling_coefficient!(
    builder, orig_var, coeff, cstr_moi, sp_var_index, sp_id_map
)
    haskey(sp_var_index, orig_var) || return
    user_id, moi_idx = sp_var_index[orig_var]
    add_coupling_coefficient!(
        builder, sp_id_map[user_id], moi_idx, cstr_moi, coeff
    )
end

function _register_pure_master_data!(
    builder, model, master_model, master_vars,
    master_constrs, constr_mapping, var_mapping,
    orig_costs
)
    isempty(master_vars) && return

    for (var_name, indexes) in master_vars
        for idx in sort(collect(indexes))
            orig_var = get_scalar_object(model, var_name, idx)
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

            add_pure_master_variable!(
                builder, moi_idx, cost, lb, ub, is_int
            )

            # Find coupling constraints involving this variable
            for (constr_name, constr_indexes) in master_constrs
                for cidx in constr_indexes
                    orig_constr = get_scalar_object(
                        model, constr_name, cidx
                    )
                    orig_constr_obj = JuMP.constraint_object(
                        orig_constr
                    )
                    coeff = _get_coeff_in_func(
                        orig_constr_obj.func, orig_var
                    )
                    if !iszero(coeff)
                        reform_constr = constr_mapping[orig_constr]
                        cstr_moi = JuMP.index(reform_constr)
                        add_pure_master_coupling!(
                            builder, moi_idx, cstr_moi, coeff
                        )
                    end
                end
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
