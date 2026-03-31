# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

"""Return all permutations of 1:n as a vector of vectors."""
function _all_permutations(n::Int)
    result = Vector{Vector{Int}}()
    _permute!(collect(1:n), 1, result)
    return result
end

function _permute!(arr, k, result)
    if k == length(arr)
        push!(result, copy(arr))
        return
    end
    for i in k:length(arr)
        arr[k], arr[i] = arr[i], arr[k]
        _permute!(arr, k + 1, result)
        arr[k], arr[i] = arr[i], arr[k]
    end
end

"""
    _for_all_permutations(verify_fn, n, initial_state)

Drive `verify_fn(current_state, idx) -> next_state` over every
permutation of `1:n`, threading the returned state across visits.
"""
function _for_all_permutations(
    verify_fn::Function,
    n::Int,
    initial_state,
)
    current = initial_state
    for perm in _all_permutations(n)
        for idx in perm
            current = verify_fn(current, idx)
        end
    end
end
