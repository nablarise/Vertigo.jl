# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    IdenticalBlock

The one element of an index set built by [`identical`](@ref): a name and the
lower and upper multiplicities, the smallest and largest number of copies of
the block a solution uses.
"""
struct IdenticalBlock
    name::Symbol
    lm::Int
    um::Int
end

Base.show(io::IO, k::IdenticalBlock) = print(io, k.name, "{", k.lm, "..", k.um, "}")

# The DML sorts subproblem ids.
function Base.isless(a::IdenticalBlock, b::IdenticalBlock)
    return isless((a.name, a.lm, a.um), (b.name, b.lm, b.um))
end

"""
    identical(name, um; lm = 0) -> Vector{IdenticalBlock}

Return an index set with one element, standing for between `lm` and `um`
identical copies of a subproblem.

Inputs:
- `name`: a `Symbol` telling this set apart from another one with the same
  multiplicities; two sets with the same name and multiplicities are the same
  subproblem
- `um`: the upper multiplicity, the largest number of copies, a finite integer
- `lm`: the lower multiplicity, the smallest number of copies, a nonnegative
  integer

Throws an `ArgumentError` unless `0 <= lm <= um < Inf` and both are integers.

The model is written for one representative copy, and a subproblem annotated
with the element of this set gets the multiplicities `(lm, um)`:

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
julia> identical(:bins, 4; lm = 1)
1-element Vector{Vertigo.DML.IdenticalBlock}:
 bins{1..4}
```
"""
function identical(name::Symbol, um; lm = 0)
    if !(_is_finite_integer(lm) && _is_finite_integer(um) && 0 <= lm <= um)
        throw(ArgumentError(
            "identical($(repr(name)), $um; lm = $lm): lm and um must be " *
            "finite integers with 0 <= lm <= um"
        ))
    end
    return [IdenticalBlock(name, Int(lm), Int(um))]
end

_is_finite_integer(n) = isfinite(n) && isinteger(n)

# The lower and upper multiplicities of a subproblem: the number of copies of
# an identical block, exactly one copy of any other subproblem.
_multiplicity(id::IdenticalBlock) = (Float64(id.lm), Float64(id.um))
_multiplicity(_) = (1.0, 1.0)
