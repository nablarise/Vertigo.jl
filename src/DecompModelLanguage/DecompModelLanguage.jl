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

include("UserProblemDescription.jl")
include("Checker.jl")
include("Mappings.jl")
include("Transition.jl")
include("Decomposition.jl")

export dantzig_wolfe_decomposition
export MasterProblemUserDescription, SubproblemUserDescription
export MasterProblemOptions, SubproblemOptions

end # module DML
