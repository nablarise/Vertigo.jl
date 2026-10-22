# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# Annotations are built by keyword only: a positional constructor made it too easy to
# pass `silent` and `user_overriden` in the wrong order (both are `Union{Nothing,Bool}`,
# so nothing would complain).
struct MasterAnnotation
    silent::Union{Nothing,Bool}

    MasterAnnotation(; silent=nothing) = new(silent)
end

# An option left `nothing` means the rule does not say; the model then gets
# the default (silent, built as a MOI model) unless another rule assigning
# to the same model asks otherwise.
struct SubproblemAnnotation
    id::Any
    user_overriden::Union{Nothing,Bool}
    silent::Union{Nothing,Bool}

    function SubproblemAnnotation(; id, user_overriden=nothing, silent=nothing)
        return new(id, user_overriden, silent)
    end
end

dantzig_wolfe_master(; silent=nothing) = MasterAnnotation(; silent)
function dantzig_wolfe_subproblem(
    id; silent=nothing, user_overriden=nothing
)
    return SubproblemAnnotation(; id, user_overriden, silent)
end
