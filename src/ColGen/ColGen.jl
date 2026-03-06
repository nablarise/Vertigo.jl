# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

module ColGen

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using Printf

using ..Reformulation
import ..Reformulation: is_minimization

# ColGen alias: coluna.jl calls ColGen.get_dual_bound(...) and ColGen.run!(...)
const ColGen = @__MODULE__

# ── Coluna kernel (DO NOT MODIFY) ─────────────────────────────────────────────
include("coluna.jl")

# ── Utilities ─────────────────────────────────────────────────────────────────
include("dw_colgen_iteration.jl")

# ── MOI solution types ───────────────────────────────────────────────────────
include("moi_solutions.jl")

# ── Helpers ──────────────────────────────────────────────────────────────────
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

# Context and solver entry point
export ColGenContext, ColGenLoggerContext, run_column_generation, ColGenOutput
export ColGenStatus, optimal, master_infeasible, subproblem_infeasible, iteration_limit
export WentgesSmoothing, NoStabilization

end # module ColGen
