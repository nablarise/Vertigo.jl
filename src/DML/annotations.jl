# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# `silent === nothing` means the rule does not say; the model is then silent
# unless another rule assigning to the same model asks otherwise.
struct SubproblemAnnotation
    id::Any
    silent::Union{Nothing,Bool}
end

struct MasterAnnotation
    silent::Union{Nothing,Bool}
end

dantzig_wolfe_subproblem(id; silent=nothing) = SubproblemAnnotation(id, silent)
dantzig_wolfe_master(; silent=nothing) = MasterAnnotation(silent)
