# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module Vertigo

# ── Reformulation (decomposition types) ──────────────────────────────────────
include("Reformulation/Reformulation.jl")
using .Reformulation

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
export PricingSubproblemId
export DWReformulation, DWReformulationBuilder
export add_subproblem!, add_sp_variable!, add_coupling_coefficient!
export add_mapping!, add_pure_master_variable!, add_pure_master_coupling!
export add_coupling_constraint!, build

# MOI type wrappers
export TaggedCI

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

# Branch-and-price
export BPSpace, BPEvaluator, BPOutput, run_branch_and_price
export TreeSearchLoggerContext

end # module Vertigo
