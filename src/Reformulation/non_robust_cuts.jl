# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct ActiveNonRobustCut{C}
    master_constraint::C
    family::AbstractNonRobustCutFamily
    data::Any
end

mutable struct NonRobustCutManager{C}
    cuts::Vector{ActiveNonRobustCut{C}}
    duals::Dict{C,Float64}
end

NonRobustCutManager{C}() where {C} = NonRobustCutManager(
    ActiveNonRobustCut{C}[], Dict{C,Float64}()
)

function add_cut!(
    mgr::NonRobustCutManager{C},
    master_cstr::C,
    family::AbstractNonRobustCutFamily,
    data
) where {C}
    push!(mgr.cuts, ActiveNonRobustCut(master_cstr, family, data))
    mgr.duals[master_cstr] = 0.0
    return nothing
end

function update_duals!(mgr::NonRobustCutManager, dual_solution)
    for cut in mgr.cuts
        mgr.duals[cut.master_constraint] = dual_solution(cut.master_constraint)
    end
    return nothing
end

function total_cut_dual_contribution(mgr::NonRobustCutManager, sp_id, sp_var)
    total = 0.0
    for cut in mgr.cuts
        σ = mgr.duals[cut.master_constraint]
        iszero(σ) && continue
        total += compute_cut_dual_contribution(cut.family, cut.data, σ, sp_id, sp_var)
    end
    return total
end

function compute_column_cut_coefficients(
    mgr::NonRobustCutManager{C}, sol::AbstractSubproblemSolution
) where {C}
    coeffs = Dict{C,Float64}()
    for cut in mgr.cuts
        coeff = compute_cut_coefficient(cut.family, cut.data, sol)
        if !iszero(coeff)
            coeffs[cut.master_constraint] = coeff
        end
    end
    return coeffs
end
