# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module ColGen

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Printf

# ColGen alias: coluna.jl calls ColGen.get_dual_bound(...) and ColGen.run!(...)
const ColGen = @__MODULE__

# ── Coluna kernel (DO NOT MODIFY) ─────────────────────────────────────────────
include("coluna.jl")

# ── Utilities ─────────────────────────────────────────────────────────────────
include("dw_colgen_iteration.jl")

# ── Decomposition interface (abstract types + function stubs) ─────────────────
include("decomposition_interface.jl")

# ── Concrete decomposition and column pool ────────────────────────────────────
include("decomposition_impl.jl")

# ── MOI solution types (needs TaggedCI from decomposition_impl.jl) ────────────
include("moi_solutions.jl")

# ── Helpers (needs TaggedCI from decomposition_impl.jl) ──────────────────────
include("helpers.jl")

# ── Context and phase/stage control ───────────────────────────────────────────
include("context.jl")

# ── Dispatch implementations ──────────────────────────────────────────────────
include("master_optimization.jl")
include("reduced_costs.jl")
include("pricing_optimization.jl")
include("column_insertion.jl")
include("dual_bounds.jl")
include("dw_stabilization.jl")
include("ip_management.jl")
include("logger.jl")

# ── Exports ───────────────────────────────────────────────────────────────────

# Decomposition builder
export Decomposition, DecompositionBuilder, ConstraintSense, EQUAL_TO, LESS_THAN, GREATER_THAN
export add_subproblem!, add_sp_variable!, add_coupling_coefficient!
export add_mapping!, add_pure_master_variable!, add_pure_master_coupling!
export add_coupling_constraint!, build

# Data structures
export ColumnPool, NonRobustCutManager

# Context and solver entry point
export ColGenContext, ColGenLoggerContext, run_column_generation, ColGenOutput
export ColGenStatus, optimal, master_infeasible, subproblem_infeasible, iteration_limit
export WentgesSmoothing, NoStabilization

# Decomposition interface query functions
export subproblem_ids, subproblem_variables, subproblem_fixed_cost, convexity_bounds
export coupling_constraints, is_minimization, original_cost, coupling_coefficients
export pure_master_variables, pure_master_cost, pure_master_bounds
export nonzero_entries, solution_value, subproblem_id
export has_column, record_column!, columns, columns_for_subproblem
export column_sp_id, column_original_cost, pricing_objective_value
export column_nonzero_entries

end # module ColGen
