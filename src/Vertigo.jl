# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module Vertigo

# ── Column generation algorithm ───────────────────────────────────────────────
include("ColGen/ColGen.jl")
using .ColGen

# ── MOI model state tracker ───────────────────────────────────────────────────
include("MathOptState/MathOptState.jl")
using .MathOptState

# ── General-purpose tree search ───────────────────────────────────────────────
include("TreeSearch/TreeSearch.jl")
using .TreeSearch

# ── Branch-cut-price ─────────────────────────────────────────────────────────
include("BranchCutPrice/BranchCutPrice.jl")
using .BranchCutPrice

# ── Adapter stub ──────────────────────────────────────────────────────────────
include("rk_adapter.jl")

# ── Public API ────────────────────────────────────────────────────────────────

# Decomposition builder
export Decomposition, DecompositionBuilder, ConstraintSense, EQUAL_TO, LESS_THAN, GREATER_THAN
export add_subproblem!, add_sp_variable!, add_coupling_coefficient!
export add_mapping!, add_pure_master_variable!, add_pure_master_coupling!
export add_coupling_constraint!, build

# Data structures
export ColumnPool, SpSolution, NonRobustCutManager

# Context and solver entry point
export ColGenContext, ColGenLoggerContext, run_column_generation, ColGenOutput
export ColGenStatus, optimal, master_infeasible, subproblem_infeasible, iteration_limit

# Decomposition interface query functions
export subproblem_ids, subproblem_variables, subproblem_fixed_cost, convexity_bounds
export coupling_constraints, is_minimization, original_cost, coupling_coefficients
export pure_master_variables, pure_master_cost, pure_master_bounds
export nonzero_entries, solution_value, subproblem_id
export has_column, record_column!, get_column_solution, columns, columns_for_subproblem

# Branch-and-price
export BPSpace, BPEvaluator, BPOutput, run_branch_and_price
export TreeSearchLoggerContext

end # module Vertigo
