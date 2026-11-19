# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

struct SubproblemAnnotation
    id::Any
    multiplicity::Union{Nothing,Tuple{Float64,Float64}}
end

struct MasterAnnotation
end

"""
    dantzig_wolfe_subproblem(id; multiplicity=nothing) -> SubproblemAnnotation

Assign a variable or a constraint to the subproblem `id`.

`multiplicity = (lb, ub)` bounds the number of solutions of the subproblem
that a solution of the problem uses, counted with repetitions; these become
the convexity constraints of the master. One subproblem for `fleet` identical
vehicles, where a solution is a route, takes `(0, fleet)`.

A subproblem whose annotations give no multiplicity gets `(1, 1)`. Annotations
of the same subproblem that give one must give the same, or the decomposition
throws an `ArgumentError`.

Throws an `ArgumentError` unless `0 <= lb <= ub < Inf`.
"""
function dantzig_wolfe_subproblem(id; multiplicity=nothing)
    if isnothing(multiplicity)
        return SubproblemAnnotation(id, nothing)
    end
    lb, ub = multiplicity
    if !(0 <= lb <= ub < Inf)
        throw(ArgumentError(
            "the multiplicity of the subproblem $id is ($lb, $ub); " *
            "it must satisfy 0 <= lb <= ub < Inf"
        ))
    end
    return SubproblemAnnotation(id, (Float64(lb), Float64(ub)))
end

dantzig_wolfe_master() = MasterAnnotation()
