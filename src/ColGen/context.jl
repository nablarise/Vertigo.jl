# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# IP PRIMAL SOLUTION
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    MasterIpPrimalSol

An IP-feasible primal solution of the restricted master problem,
expressed in terms of master variables (columns and pure master
variables).

# Fields
- `obj_value`: objective value of the solution.
- `non_zero_integral`: integer variables (columns or pure master)
  with nonzero multiplicity.
- `non_zero_continuous`: continuous pure master variables with
  nonzero value.
"""
struct MasterIpPrimalSol
    obj_value::Float64
    non_zero_integral::Vector{Tuple{MOI.VariableIndex,Int}}
    non_zero_continuous::Vector{Tuple{MOI.VariableIndex,Float64}}
end

function MasterIpPrimalSol(
    obj_value::Float64,
    non_zero_integral::Vector{Tuple{MOI.VariableIndex,Int}}
)
    return MasterIpPrimalSol(
        obj_value, non_zero_integral,
        Tuple{MOI.VariableIndex,Float64}[]
    )
end

# ────────────────────────────────────────────────────────────────────────────────────────
# MASTER WRAPPER
# ────────────────────────────────────────────────────────────────────────────────────────

struct Master{MoiModel}
    moi_master::MoiModel
    convexity_constraints_ub::Dict{Any,Any}
    convexity_constraints_lb::Dict{Any,Any}
    eq_art_vars::Dict{Any,Any}
    leq_art_vars::Dict{Any,Any}
    geq_art_vars::Dict{Any,Any}
end

moi_master(m::Master) = m.moi_master

# ────────────────────────────────────────────────────────────────────────────────────────
# PRICING SUBPROBLEM WRAPPER
# ────────────────────────────────────────────────────────────────────────────────────────

struct PricingSubproblem{M}
    moi_model::M
end

moi_pricing_sp(sp::PricingSubproblem) = sp.moi_model

# ────────────────────────────────────────────────────────────────────────────────────────
# ACTIVE BRANCHING CONSTRAINT
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    ActiveBranchingConstraint

A branching constraint currently active in the master LP. Maps a MOI constraint
(added via `LocalCutTracker`) to the original variable it branches on.
"""
struct ActiveBranchingConstraint
    constraint_index::Any
    orig_var::Any
end

# ────────────────────────────────────────────────────────────────────────────────────────
# COLGEN CONTEXT
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    ColGenContext{D,M,P,CutM}

Column generation context. Bundles all static data (decomposition), mutable MOI models,
and runtime structures (column pool, cut manager, artificial variable tracking).

Type parameters:
  - D: AbstractDecomposition implementation
  - M: master MOI backend type
  - P: ColumnPool type
  - CutM: NonRobustCutManager type
"""
mutable struct ColGenContext{D<:AbstractDecomposition,M,P<:ColumnPool,CutM<:NonRobustCutManager}
    decomp::D
    master_model::M
    convexity_ub::Dict{Any,Any}   # sp_id → MOI.ConstraintIndex (LessThan)
    convexity_lb::Dict{Any,Any}   # sp_id → MOI.ConstraintIndex (GreaterThan)
    sp_models::Dict{Any,Any}      # sp_id → MOI backend
    pool::P
    cuts::CutM
    eq_art_vars::Dict{Any,Any}    # cstr_idx → (MOI.VariableIndex, MOI.VariableIndex)
    leq_art_vars::Dict{Any,Any}   # cstr_idx → MOI.VariableIndex
    geq_art_vars::Dict{Any,Any}   # cstr_idx → MOI.VariableIndex
    ip_incumbent::Union{Nothing,MasterIpPrimalSol}
    ip_primal_bound::Union{Nothing,Float64}
    branching_constraints::Vector{ActiveBranchingConstraint}
    smoothing_alpha::Float64

    function ColGenContext(
        decomp, master_model, convexity_ub, convexity_lb, sp_models,
        pool, cuts, eq_art_vars, leq_art_vars, geq_art_vars;
        smoothing_alpha::Float64 = 0.0
    )
        new{typeof(decomp),typeof(master_model),typeof(pool),typeof(cuts)}(
            decomp, master_model, convexity_ub, convexity_lb, sp_models,
            pool, cuts, eq_art_vars, leq_art_vars, geq_art_vars, nothing,
            nothing, ActiveBranchingConstraint[], smoothing_alpha
        )
    end
end

# Core accessors
is_minimization(ctx::ColGenContext) = is_minimization(ctx.decomp)

function get_master(ctx::ColGenContext)
    return Master(
        ctx.master_model,
        ctx.convexity_ub,
        ctx.convexity_lb,
        ctx.eq_art_vars,
        ctx.leq_art_vars,
        ctx.geq_art_vars
    )
end

function get_pricing_subprobs(ctx::ColGenContext)
    return Dict{Any,Any}(
        sp_id => PricingSubproblem(ctx.sp_models[sp_id])
        for sp_id in subproblem_ids(ctx.decomp)
    )
end

get_reform(ctx::ColGenContext) = ctx

function _dual_bound_dominated(ctx, dual_bound, ip_bound)
    isnothing(ip_bound) && return false
    isnothing(dual_bound) && return false
    if is_minimization(ctx)
        return dual_bound >= ip_bound - 1e-6
    else
        return dual_bound <= ip_bound + 1e-6
    end
end


# ────────────────────────────────────────────────────────────────────────────────────────
# PHASE / STAGE TYPES
# ────────────────────────────────────────────────────────────────────────────────────────

struct ColGenPhaseIterator end
struct ColGenStageIterator end

struct Phase0
    artificial_var_cost::Float64
    convexity_artificial_var_cost::Float64
    Phase0(ac=10000.0, cc=10000.0) = new(ac, cc)
end

struct Phase1 end   # Minimise sum of artificial variables (feasibility)
struct Phase2 end   # Optimise original objective, no artificial variables

struct ExactStage end
struct NoStabilization end

@enum ColGenStatus optimal master_infeasible subproblem_infeasible iteration_limit

new_phase_iterator(::ColGenContext) = ColGenPhaseIterator()
initial_phase(::ColGenPhaseIterator) = Phase0()
new_stage_iterator(::ColGenContext) = ColGenStageIterator()
initial_stage(::ColGenStageIterator) = ExactStage()

stop_colgen(::ColGenContext, ::Nothing) = false
stop_colgen(::ColGenContext, _) = false


# ────────────────────────────────────────────────────────────────────────────────────────
# SETUP REFORMULATION (phase 1 artificial variables + integrality relaxation)
# ────────────────────────────────────────────────────────────────────────────────────────

function setup_reformulation!(ctx::ColGenContext, phase::Phase0)
    model = ctx.master_model
    sense = is_minimization(ctx.decomp) ? 1 : -1
    cost = sense * phase.artificial_var_cost
    convexity_cost = sense * phase.convexity_artificial_var_cost

    # Artificial variables for coupling constraints
    for (cstr_id, cstr_sense, _rhs) in coupling_constraints(ctx.decomp)
        if cstr_sense == EQUAL_TO
            s_pos = add_variable!(model;
                lower_bound = 0.0,
                constraint_coeffs = Dict(cstr_id => 1.0),
                objective_coeff = cost,
                name = "s_pos[$(cstr_id.value)]"
            )
            s_neg = add_variable!(model;
                lower_bound = 0.0,
                constraint_coeffs = Dict(cstr_id => -1.0),
                objective_coeff = cost,
                name = "s_neg[$(cstr_id.value)]"
            )
            ctx.eq_art_vars[cstr_id] = (s_pos, s_neg)
        elseif cstr_sense == GREATER_THAN
            s_pos = add_variable!(model;
                lower_bound = 0.0,
                constraint_coeffs = Dict(cstr_id => 1.0),
                objective_coeff = cost,
                name = "s_geq[$(cstr_id.value)]"
            )
            ctx.geq_art_vars[cstr_id] = s_pos
        elseif cstr_sense == LESS_THAN
            s_neg = add_variable!(model;
                lower_bound = 0.0,
                constraint_coeffs = Dict(cstr_id => -1.0),
                objective_coeff = cost,
                name = "s_leq[$(cstr_id.value)]"
            )
            ctx.leq_art_vars[cstr_id] = s_neg
        end
    end

    # Artificial variables for convexity constraints (LessThan UB)
    for (sp_id, cstr_idx) in ctx.convexity_ub
        s_neg = add_variable!(model;
            lower_bound = 0.0,
            constraint_coeffs = Dict(cstr_idx => -1.0),
            objective_coeff = convexity_cost,
            name = "s_conv_ub[$(sp_id)]"
        )
        ctx.leq_art_vars[cstr_idx] = s_neg
    end

    # Artificial variables for convexity constraints (GreaterThan LB)
    for (sp_id, cstr_idx) in ctx.convexity_lb
        s_pos = add_variable!(model;
            lower_bound = 0.0,
            constraint_coeffs = Dict(cstr_idx => 1.0),
            objective_coeff = convexity_cost,
            name = "s_conv_lb[$(sp_id)]"
        )
        ctx.geq_art_vars[cstr_idx] = s_pos
    end

    # Relax integrality: delete Integer constraints
    integer_constraints = MOI.get(
        model,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}()
    )
    for ci in integer_constraints
        MOI.delete(model, ci)
    end

    # Relax integrality: delete ZeroOne (binary) constraints, ensure [0,1] bounds
    binary_constraints = MOI.get(
        model,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}()
    )
    for ci in binary_constraints
        var_idx = MOI.get(model, MOI.ConstraintFunction(), ci)
        MOI.delete(model, ci)
        ub_ci = MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}}(var_idx.value)
        if !MOI.is_valid(model, ub_ci)
            MOI.add_constraint(model, var_idx, MOI.LessThan(1.0))
        end
        lb_ci = MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}(var_idx.value)
        if !MOI.is_valid(model, lb_ci)
            MOI.add_constraint(model, var_idx, MOI.GreaterThan(0.0))
        end
    end

    return nothing
end

function setup_reformulation!(ctx::ColGenContext, ::Phase1)
    model = ctx.master_model
    obj_type = MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
    sense = is_minimization(ctx.decomp) ? 1.0 : -1.0

    for (master_var, _, _, _) in columns(ctx.pool)
        MOI.modify(model, obj_type, MOI.ScalarCoefficientChange(master_var, 0.0))
    end
    for pmv in pure_master_variables(ctx.decomp)
        MOI.modify(model, obj_type, MOI.ScalarCoefficientChange(pmv.id, 0.0))
    end
    for (_, (s_pos, s_neg)) in ctx.eq_art_vars
        MOI.modify(model, obj_type, MOI.ScalarCoefficientChange(s_pos, sense))
        MOI.modify(model, obj_type, MOI.ScalarCoefficientChange(s_neg, sense))
    end
    for (_, s) in ctx.leq_art_vars
        MOI.modify(model, obj_type, MOI.ScalarCoefficientChange(s, sense))
    end
    for (_, s) in ctx.geq_art_vars
        MOI.modify(model, obj_type, MOI.ScalarCoefficientChange(s, sense))
    end
    return nothing
end

function setup_reformulation!(ctx::ColGenContext, ::Phase2)
    model = ctx.master_model
    obj_type = MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()

    for (_, (s_pos, s_neg)) in ctx.eq_art_vars
        MOI.delete(model, s_pos)
        MOI.delete(model, s_neg)
    end
    for (_, s) in ctx.leq_art_vars
        MOI.delete(model, s)
    end
    for (_, s) in ctx.geq_art_vars
        MOI.delete(model, s)
    end
    empty!(ctx.eq_art_vars)
    empty!(ctx.leq_art_vars)
    empty!(ctx.geq_art_vars)

    for (master_var, _, _, cost) in columns(ctx.pool)
        MOI.modify(model, obj_type, MOI.ScalarCoefficientChange(master_var, cost))
    end
    for pmv in pure_master_variables(ctx.decomp)
        MOI.modify(model, obj_type,
            MOI.ScalarCoefficientChange(pmv.id, pure_master_cost(ctx.decomp, pmv)))
    end
    return nothing
end


# ────────────────────────────────────────────────────────────────────────────────────────
# COLGEN ITERATION OUTPUT
# ────────────────────────────────────────────────────────────────────────────────────────

struct ColGenIterationOutput
    master_lp_obj::Union{Float64,Nothing}
    dual_bound::Union{Float64,Nothing}
    nb_columns_added::Int64
    master_lp_primal_sol::Any
    master_ip_primal_sol::Any
    subproblem_infeasible::Bool
end

colgen_iteration_output_type(::ColGenContext) = ColGenIterationOutput

stop_colgen_phase(::ColGenContext, _, ::Nothing, _, _, _) = false

function stop_colgen_phase(
    ctx::ColGenContext,
    ::Union{Phase0,Phase2},
    colgen_iter_output::ColGenIterationOutput,
    incumbent_dual_bound,
    ip_primal_sol,
    iteration
)
    master_lp_obj = colgen_iter_output.master_lp_obj
    no_column_added = colgen_iter_output.nb_columns_added == 0
    iteration_limit = iteration > 1000
    lp_gap_closed = (
        !isnothing(master_lp_obj) &&
        !isnothing(incumbent_dual_bound) &&
        abs(master_lp_obj - incumbent_dual_bound) < 1e-6
    )
    ip_pruned = _dual_bound_dominated(
        ctx, incumbent_dual_bound, ctx.ip_primal_bound
    )
    return iteration_limit || no_column_added ||
           lp_gap_closed || ip_pruned
end

function stop_colgen_phase(
    ctx::ColGenContext,
    ::Phase1,
    colgen_iter_output::ColGenIterationOutput,
    incumbent_dual_bound,
    ip_primal_sol,
    iteration
)
    # Phase1 has no iteration limit — feasibility must reach true convergence
    master_lp_obj = colgen_iter_output.master_lp_obj
    no_column_added = colgen_iter_output.nb_columns_added == 0
    lp_gap_closed = (
        !isnothing(master_lp_obj) &&
        !isnothing(incumbent_dual_bound) &&
        abs(master_lp_obj - incumbent_dual_bound) < 1e-6
    )
    return no_column_added || lp_gap_closed
end

function new_iteration_output(
    ::Type{<:ColGenIterationOutput},
    min_sense,
    mlp,
    db,
    nb_new_cols,
    new_cut_in_master,
    infeasible_master,
    unbounded_master,
    infeasible_subproblem,
    unbounded_subproblem,
    time_limit_reached,
    master_lp_primal_sol,
    master_ip_primal_sol,
    master_lp_dual_sol,
)
    return ColGenIterationOutput(mlp, db, nb_new_cols, master_lp_dual_sol, master_ip_primal_sol, infeasible_subproblem)
end

get_dual_bound(output::ColGenIterationOutput) = output.dual_bound


# ────────────────────────────────────────────────────────────────────────────────────────
# COLGEN PHASE OUTPUT
# ────────────────────────────────────────────────────────────────────────────────────────

struct ColGenPhaseOutput
    master_lp_obj::Union{Nothing,Float64}
    incumbent_dual_bound::Union{Nothing,Float64}
    nb_iterations::Int
    has_artificial_vars::Bool   # condition A: art vars active in solution
    colgen_converged::Bool      # condition D: CG has converged
    subproblem_infeasible::Bool
end

colgen_phase_output_type(::ColGenContext) = ColGenPhaseOutput

function has_artificial_vars_in_solution(ctx::ColGenContext, tol=1e-6)::Bool
    model = ctx.master_model
    for (_, (s_pos, s_neg)) in ctx.eq_art_vars
        MOI.get(model, MOI.VariablePrimal(), s_pos) > tol && return true
        MOI.get(model, MOI.VariablePrimal(), s_neg) > tol && return true
    end
    for (_, s) in ctx.leq_art_vars
        MOI.get(model, MOI.VariablePrimal(), s) > tol && return true
    end
    for (_, s) in ctx.geq_art_vars
        MOI.get(model, MOI.VariablePrimal(), s) > tol && return true
    end
    return false
end

function new_phase_output(
    ::Type{<:ColGenPhaseOutput},
    ctx::ColGenContext,
    min_sense,
    phase,
    stage,
    colgen_iter_output::ColGenIterationOutput,
    iteration,
    inc_dual_bound
)
    mlp = colgen_iter_output.master_lp_obj
    lp_gap_closed = (
        !isnothing(mlp) && !isnothing(inc_dual_bound) &&
        abs(mlp - inc_dual_bound) < 1e-6
    )
    subprob_inf = colgen_iter_output.subproblem_infeasible
    converged   = lp_gap_closed || (colgen_iter_output.nb_columns_added == 0 && !subprob_inf)
    has_art = has_artificial_vars_in_solution(ctx)
    return ColGenPhaseOutput(mlp, inc_dual_bound, iteration, has_art, converged, subprob_inf)
end

function next_phase(::ColGenPhaseIterator, ::Phase0, o::ColGenPhaseOutput)
    o.subproblem_infeasible && return nothing
    o.has_artificial_vars   && return Phase1()
    o.colgen_converged      && return nothing
    return Phase2()
end

function next_phase(::ColGenPhaseIterator, ::Phase1, o::ColGenPhaseOutput)
    o.subproblem_infeasible  && return nothing
    !o.has_artificial_vars   && return Phase2()
    o.colgen_converged       && return nothing   # infeasible confirmed
    return Phase1()
end

function next_phase(::ColGenPhaseIterator, ::Phase2, o::ColGenPhaseOutput)
    o.subproblem_infeasible && return nothing
    o.has_artificial_vars   && error("Artificial variables detected in Phase2")
    o.colgen_converged      && return nothing
    return nothing   # iteration limit — stop rather than loop
end

function next_stage(::ColGenStageIterator, ::ExactStage, ::ColGenPhaseOutput)
    return ExactStage()
end


# ────────────────────────────────────────────────────────────────────────────────────────
# COLGEN OUTPUT
# ────────────────────────────────────────────────────────────────────────────────────────

struct ColGenOutput
    status::ColGenStatus
    master_lp_obj::Union{Nothing,Float64}
    incumbent_dual_bound::Union{Nothing,Float64}
    ip_incumbent::Union{Nothing,MasterIpPrimalSol}
end

colgen_output_type(::ColGenContext) = ColGenOutput

function new_output(
    ::Type{ColGenOutput}, ctx::ColGenContext, p::ColGenPhaseOutput
)
    ip = ctx.ip_incumbent
    if p.subproblem_infeasible
        return ColGenOutput(
            subproblem_infeasible, nothing, nothing, ip
        )
    elseif p.has_artificial_vars && p.colgen_converged
        return ColGenOutput(
            master_infeasible, nothing, nothing, ip
        )
    elseif p.colgen_converged
        return ColGenOutput(
            optimal, p.master_lp_obj,
            p.incumbent_dual_bound, ip
        )
    else
        return ColGenOutput(
            iteration_limit, p.master_lp_obj,
            p.incumbent_dual_bound, ip
        )
    end
end


# ────────────────────────────────────────────────────────────────────────────────────────
# ITERATION LOGGING
# ────────────────────────────────────────────────────────────────────────────────────────

function after_colgen_iteration(
    ::ColGenContext,
    ::Union{Phase0,Phase1,Phase2},
    ::ExactStage,
    colgen_iterations::Int64,
    ::NoStabilization,
    colgen_iter_output::ColGenIterationOutput,
    incumbent_dual_bound,
    ip_primal_sol
)
    # do nothing
end

function is_better_dual_bound(ctx::ColGenContext, dual_bound::Float64, incumbent::Float64)
    sense = is_minimization(ctx) ? 1 : -1
    return sense * dual_bound > sense * incumbent
end


# ────────────────────────────────────────────────────────────────────────────────────────
# RUN COLUMN GENERATION ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    run_column_generation(ctx::ColGenContext) -> ColGenOutput

Run the column generation algorithm on the given context.

# Examples
```jldoctest
julia> # (see test for a full example)
```
"""
function run_column_generation(ctx::ColGenContext)
    return ColGen.run!(ctx, nothing)
end
