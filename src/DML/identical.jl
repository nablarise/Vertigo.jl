# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    IdenticalBlock

The one element of an index set built by [`identical`](@ref): a name and the
bounds on the number of copies of the block a solution uses.
"""
struct IdenticalBlock
    name::Symbol
    lb::Int
    ub::Int
end

Base.show(io::IO, k::IdenticalBlock) = print(io, k.name, "{", k.lb, "..", k.ub, "}")

# The DML sorts subproblem ids.
function Base.isless(a::IdenticalBlock, b::IdenticalBlock)
    return isless((a.name, a.lb, a.ub), (b.name, b.lb, b.ub))
end

"""
    identical(name, ub; lb = 0) -> Vector{IdenticalBlock}

Return an index set with one element, standing for between `lb` and `ub`
identical copies of a subproblem.

Inputs:
- `name`: a `Symbol` telling this set apart from another one with the same
  bounds; two sets with the same name and bounds are the same subproblem
- `ub`: the largest number of copies, a finite integer
- `lb`: the smallest number of copies, a nonnegative integer

Throws an `ArgumentError` unless `0 <= lb <= ub < Inf` and both are integers.

The model is written for one representative copy, and a subproblem annotated
with the element of this set gets the convexity bounds `(lb, ub)`:

```julia
K = identical(:bins, nb_items)
@variable(model, x[k in K, i in items], Bin)
@constraint(model, cover[i in items], sum(x[k, i] for k in K) >= 1)

decomp, sp_map = @dantzig_wolfe model begin
    x[k, _] => subproblem(k)
    cover[_] => master()
end
```

In the master, `sum(x[k, i] for k in K)` is the sum over all the copies used.
JuMP sees one copy only: `optimize!(model)` solves a problem with a single
bin, which is not the problem the decomposition solves.

# Examples

```jldoctest
julia> identical(:bins, 4; lb = 1)
1-element Vector{Vertigo.DML.IdenticalBlock}:
 bins{1..4}
```
"""
function identical(name::Symbol, ub; lb = 0)
    if !(_is_finite_integer(lb) && _is_finite_integer(ub) && 0 <= lb <= ub)
        throw(ArgumentError(
            "identical($(repr(name)), $ub; lb = $lb): lb and ub must be " *
            "finite integers with 0 <= lb <= ub"
        ))
    end
    return [IdenticalBlock(name, Int(lb), Int(ub))]
end

_is_finite_integer(n) = isfinite(n) && isinteger(n)

# The convexity bounds of a subproblem: the number of copies of an identical
# block, exactly one copy of any other subproblem.
_convexity_bounds(id::IdenticalBlock) = (Float64(id.lb), Float64(id.ub))
_convexity_bounds(_) = (1.0, 1.0)
