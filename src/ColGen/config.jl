# Copyright (c) 2026 Nablarise. All rights reserved.
# Originally created by: Guillaume Marques <guillaume@nablarise.com>
# Project: Vertigo.jl
# Date: 2026-04
# Generated with: Claude Opus 4.6
# SPDX-License-Identifier: MIT

"""
    ColGenConfig

Plain data holder for user-defined column generation parameters.
Cheap to construct, copy, and compare.

# Fields
- `smoothing_alpha`: Wentges smoothing coefficient (0.0 = no smoothing).
- `max_cg_iterations`: hard limit on CG iterations per phase.
- `silent`: if `true`, skip logging (default: `false`).
"""
struct ColGenConfig
    smoothing_alpha::Float64
    max_cg_iterations::Int
    silent::Bool
    function ColGenConfig(;
        smoothing_alpha::Float64 = 0.0,
        max_cg_iterations::Int = 1000,
        silent::Bool = false
    )
        new(smoothing_alpha, max_cg_iterations, silent)
    end
end
