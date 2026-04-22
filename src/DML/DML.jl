# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    DML — Decomposition Modeling Language

High-level annotation-based syntax for Dantzig-Wolfe decomposition.
Partitions a JuMP model into master + subproblems and builds a
`DWReformulation` automatically.
"""
module DML

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using ..Reformulation

include("annotations.jl")
include("indexes.jl")
include("partition.jl")
include("models.jl")
include("bridge.jl")
include("macro.jl")

export @dantzig_wolfe
export dantzig_wolfe_decomposition
export dantzig_wolfe_subproblem, dantzig_wolfe_master

end # module DML
