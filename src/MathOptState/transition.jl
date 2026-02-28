# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""
    make_transition_callbacks(backend, helper)

Create a pair of callbacks `(apply_fwd!, apply_bwd!)` for use with
`TreeSearch.transition_to!`. Each callback applies a diff to the MOI backend.

# Arguments
- `backend`: The MOI model backend.
- `helper`: The DomainChangeTrackerHelper (or similar helper).

# Returns
A tuple `(apply_fwd!, apply_bwd!)` where each is a function `diff -> nothing`.

# Example
```julia
fwd!, bwd! = MathOptState.make_transition_callbacks(backend, helper)
TreeSearch.transition_to!(current_node, target_node, fwd!, bwd!)
```
"""
function make_transition_callbacks(backend, helper)
    apply_fwd! = diff -> apply_change!(backend, diff, helper)
    apply_bwd! = diff -> apply_change!(backend, diff, helper)
    return (apply_fwd!, apply_bwd!)
end

"""
    make_transition_callbacks(backend, helpers::Tuple)

Create callbacks for composite diffs (tuples of diffs applied to multiple helpers).

When a node carries a tuple of diffs `(domain_diff, cut_diff, fix_diff)`,
each element is applied with its corresponding helper.

# Arguments
- `backend`: The MOI model backend.
- `helpers`: A tuple of helpers, one per diff type.

# Example
```julia
helpers = (domain_helper, cut_helper, fix_helper)
fwd!, bwd! = MathOptState.make_transition_callbacks(backend, helpers)
# Now transition_to! works with nodes whose diffs are tuples
```
"""
function make_transition_callbacks(backend, helpers::Tuple)
    apply_fwd! = function(diffs::Tuple)
        for (diff, h) in zip(diffs, helpers)
            apply_change!(backend, diff, h)
        end
    end
    apply_bwd! = function(diffs::Tuple)
        for (diff, h) in zip(diffs, helpers)
            apply_change!(backend, diff, h)
        end
    end
    return (apply_fwd!, apply_bwd!)
end
