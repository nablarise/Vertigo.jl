# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module Reformulation

using MathOptInterface
const MOI = MathOptInterface

include("interface.jl")
include("moi_types.jl")
include("subproblem_data.jl")
include("dw_reformulation.jl")
include("subproblem_solution.jl")
include("column_pool.jl")
include("builder.jl")
include("non_robust_cuts.jl")

# Abstract types
export AbstractDecomposition, AbstractSubproblemSolution
export AbstractColumnPool, AbstractNonRobustCutFamily

# MOI types
export PricingSubproblemId, _SAF, _VI
export CIKind, SAF_EQ, SAF_LEQ, SAF_GEQ, VI_EQ, VI_LEQ, VI_GEQ
export TaggedCI, with_typed_ci, _ci_type
export CouplingEntry, CouplingCoefficients, get_coefficients

# Subproblem data
export SubproblemData, PureMasterVariableData, ForwardMapping

# DWReformulation
export DWReformulation
export master_model, sp_model, sp_models
export convexity_ub_ci, convexity_lb_ci
export has_convexity_ub, has_convexity_lb
export convexity_ub_pairs, convexity_lb_pairs
export set_models!

# Subproblem solution
export _SpSolution

# Column pool
export ColumnRecord, ColumnPool

# Builder
export DWReformulationBuilder
export add_subproblem!, add_sp_variable!, add_coupling_coefficient!
export add_mapping!, add_pure_master_variable!, add_pure_master_coupling!
export add_coupling_constraint!, build

# Non-robust cuts
export ActiveNonRobustCut, NonRobustCutManager
export add_cut!, update_duals!, total_cut_dual_contribution
export compute_column_cut_coefficients

# Interface functions
export original_cost, coupling_coefficients
export mapped_subproblem_variables, original_variables, mapping_to_original
export subproblem_ids, subproblem_variables, subproblem_fixed_cost
export convexity_bounds, nb_subproblem_multiplicity
export pure_master_variables, pure_master_cost, pure_master_bounds
export pure_master_is_integer, pure_master_coupling_coefficients
export coupling_constraints, is_minimization
export solution_value, nonzero_entries, subproblem_id, objective_value
export record_column!, get_column_solution, get_column_sp_id, get_column_cost
export columns, columns_for_subproblem, has_column
export compute_cut_coefficient, compute_cut_dual_contribution, is_robust
export propagate_bounds!, is_column_proper
export column_sp_id, column_original_cost, pricing_objective_value
export column_nonzero_entries

# Derived computations
export compute_column_original_cost, compute_column_coupling_coefficients
export compute_sp_reduced_costs, compute_column_reduced_cost
export compute_dual_bound_pure_master_contribution
export compute_branching_column_coefficient, project_to_original

end # module Reformulation
