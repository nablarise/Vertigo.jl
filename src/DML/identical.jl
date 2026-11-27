# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

"""
    IdenticalBlock

A subproblem standing for several identical copies, built by
[`identical`](@ref): a name and the lower and upper multiplicities, the
smallest and largest number of copies a solution uses.

It is an index set with one element, itself: `for k in K` gives `K`, so
`x[k, i]` for `k in K` and `x[K, i]` are the same variable. It stands for all
its copies, as an index set and as an index.
"""
struct IdenticalBlock
    name::Symbol
    lm::Int
    um::Int
end

Base.show(io::IO, k::IdenticalBlock) = print(io, k.name, "{", k.lm, "..", k.um, "}")

# An iterable, not an `AbstractVector`: JuMP would read `x[k, i]` with an
# `AbstractVector` `k` as a slice and return an array instead of a variable.
Base.iterate(k::IdenticalBlock) = (k, nothing)
Base.iterate(::IdenticalBlock, ::Nothing) = nothing
Base.length(::IdenticalBlock) = 1
Base.eltype(::Type{IdenticalBlock}) = IdenticalBlock

# The DML sorts subproblem ids.
function Base.isless(a::IdenticalBlock, b::IdenticalBlock)
    return isless((a.name, a.lm, a.um), (b.name, b.lm, b.um))
end

"""
    identical(name, um; lm = 0) -> IdenticalBlock

Return a subproblem standing for between `lm` and `um` identical copies, to
use as an index set.

Inputs:
- `name`: a `Symbol` telling this block apart from another one with the same
  multiplicities; two blocks with the same name and multiplicities are the
  same subproblem
- `um`: the upper multiplicity, the largest number of copies, a finite integer
- `lm`: the lower multiplicity, the smallest number of copies, a nonnegative
  integer

Throws an `ArgumentError` unless `0 <= lm <= um < Inf` and both are integers.

The model is written for one representative copy, and a subproblem annotated
with the block gets the multiplicities `(lm, um)`:

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

Several kinds of copies are a vector of blocks:
`types = [identical(:big, 2), identical(:small, 1)]`.

# Examples

```jldoctest
julia> identical(:bins, 4; lm = 1)
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
    return IdenticalBlock(name, Int(lm), Int(um))
end

_is_finite_integer(n) = isfinite(n) && isinteger(n)

# The lower and upper multiplicities of a subproblem: the number of copies of
# an identical block, exactly one copy of any other subproblem.
_multiplicity(id::IdenticalBlock) = (Float64(id.lm), Float64(id.um))
_multiplicity(_) = (1.0, 1.0)
