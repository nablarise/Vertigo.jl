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
