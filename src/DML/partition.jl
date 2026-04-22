# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

function _master_variables(model, dw_annotation)
    master_vars = Dict{Symbol,Any}()
    for (var_name, var_obj) in JuMP.object_dictionary(model)
        if var_obj isa AbstractArray && length(var_obj) > 0 && first(var_obj) isa JuMP.AbstractVariableRef
            for idx in _eachindex(var_obj)
                annotation = dw_annotation(Val(var_name), Tuple(idx)...)
                if annotation isa MasterAnnotation
                    if !haskey(master_vars, var_name)
                        master_vars[var_name] = Set{Tuple}()
                    end
                    push!(master_vars[var_name], Tuple(idx))
                end
            end
        elseif var_obj isa JuMP.AbstractVariableRef
            annotation = dw_annotation(Val(var_name))
            if annotation isa MasterAnnotation
                if !haskey(master_vars, var_name)
                    master_vars[var_name] = Set{Tuple{}}([()])
                end
            end
        end
    end
    return master_vars
end

function _partition_subproblem_variables(model, dw_annotation)
    sp_vars = Dict{Any,Dict{Symbol,Any}}()
    for (var_name, var_obj) in JuMP.object_dictionary(model)
        if var_obj isa AbstractArray && length(var_obj) > 0 && first(var_obj) isa JuMP.AbstractVariableRef
            for idx in _eachindex(var_obj)
                annotation = dw_annotation(Val(var_name), Tuple(idx)...)
                if annotation isa SubproblemAnnotation
                    sp_dict = get!(Dict{Symbol,Any}, sp_vars, annotation.id)
                    idx_set = get!(Set{Tuple}, sp_dict, var_name)
                    push!(idx_set, Tuple(idx))
                end
            end
        elseif var_obj isa JuMP.AbstractVariableRef
            annotation = dw_annotation(Val(var_name))
            if annotation isa SubproblemAnnotation
                sp_dict = get!(Dict{Symbol,Any}, sp_vars, annotation.id)
                if !haskey(sp_dict, var_name)
                    sp_dict[var_name] = Set{Tuple{}}([()])
                end
            end
        end
    end
    return sp_vars
end

function _master_constraints(model, dw_annotation)
    master_constrs = Dict{Symbol,Any}()
    for (constr_name, constr_obj) in JuMP.object_dictionary(model)
        if constr_obj isa AbstractArray && length(constr_obj) > 0 && first(constr_obj) isa JuMP.ConstraintRef
            for idx in _eachindex(constr_obj)
                annotation = dw_annotation(Val(constr_name), Tuple(idx)...)
                if annotation isa MasterAnnotation
                    if !haskey(master_constrs, constr_name)
                        master_constrs[constr_name] = Set{Tuple}()
                    end
                    push!(master_constrs[constr_name], Tuple(idx))
                end
            end
        elseif constr_obj isa JuMP.ConstraintRef
            annotation = dw_annotation(Val(constr_name))
            if annotation isa MasterAnnotation
                if !haskey(master_constrs, constr_name)
                    master_constrs[constr_name] = Set{Tuple{}}([()])
                end
            end
        end
    end
    return master_constrs
end

function _partition_subproblem_constraints(model, dw_annotation)
    sp_constrs = Dict{Any,Dict{Symbol,Any}}()
    for (constr_name, constr_obj) in JuMP.object_dictionary(model)
        if constr_obj isa AbstractArray && length(constr_obj) > 0 && first(constr_obj) isa JuMP.ConstraintRef
            for idx in _eachindex(constr_obj)
                annotation = dw_annotation(Val(constr_name), Tuple(idx)...)
                if annotation isa SubproblemAnnotation
                    sp_dict = get!(Dict{Symbol,Any}, sp_constrs, annotation.id)
                    idx_set = get!(Set{Tuple}, sp_dict, constr_name)
                    push!(idx_set, Tuple(idx))
                end
            end
        elseif constr_obj isa JuMP.ConstraintRef
            annotation = dw_annotation(Val(constr_name))
            if annotation isa SubproblemAnnotation
                sp_dict = get!(Dict{Symbol,Any}, sp_constrs, annotation.id)
                if !haskey(sp_dict, constr_name)
                    sp_dict[constr_name] = Set{Tuple{}}([()])
                end
            end
        end
    end
    return sp_constrs
end
