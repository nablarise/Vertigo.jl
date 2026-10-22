# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# An option left `nothing` means the rule does not say; the model then gets
# the default (silent, built as a MOI model) unless another rule assigning
# to the same model asks otherwise.
struct SubproblemAnnotation
    id::Any
    silent::Union{Nothing,Bool}
    user_overriden::Union{Nothing,Bool}
end

struct MasterAnnotation
    silent::Union{Nothing,Bool}
end

function dantzig_wolfe_subproblem(
    id; silent=nothing, user_overriden=nothing
)
    return SubproblemAnnotation(id, silent, user_overriden)
end
dantzig_wolfe_master(; silent=nothing) = MasterAnnotation(silent)
