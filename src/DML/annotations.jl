# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

struct SubproblemAnnotation
    id::Any
    multiplicity::Tuple{Float64,Float64}
end

struct MasterAnnotation
end

"""
    dantzig_wolfe_subproblem(id; multiplicity=(1, 1)) -> SubproblemAnnotation

Assign a variable or a constraint to the subproblem `id`.

`multiplicity = (lb, ub)` bounds the number of solutions of the subproblem
that a solution of the problem uses, counted with repetitions; these become
the convexity constraints of the master. One subproblem for `fleet` identical
vehicles, where a solution is a route, takes `(0, fleet)`.

Every annotation of the same subproblem must give the same multiplicity, or
the decomposition throws an `ArgumentError`.

Throws an `ArgumentError` unless `0 <= lb <= ub < Inf`.
"""
function dantzig_wolfe_subproblem(id; multiplicity=(1, 1))
    lb, ub = multiplicity
    0 <= lb <= ub < Inf || throw(ArgumentError(
        "the multiplicity of the subproblem $id is ($lb, $ub); " *
        "it must satisfy 0 <= lb <= ub < Inf"
    ))
    return SubproblemAnnotation(id, multiplicity)
end

dantzig_wolfe_master() = MasterAnnotation()
