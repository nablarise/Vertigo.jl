# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    DML — Decomposition Modeling Language

High-level declarative syntax for Dantzig-Wolfe decomposition.
Partitions a JuMP model into master + subproblems and builds a
`DWReformulation` automatically.
"""
module DML

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using ..Reformulation

include("indexes.jl")
include("UserProblemDescription.jl")
include("partition.jl")
include("models.jl")
include("BridgeJuMPtoInternal.jl")
include("macros/dantzig_wolfe.jl")

export @dantzig_wolfe
export dantzig_wolfe_decomposition

end # module DML
