# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct SubproblemAnnotation
    id::Any
end

struct MasterAnnotation
end

dantzig_wolfe_subproblem(id) = SubproblemAnnotation(id)
dantzig_wolfe_master() = MasterAnnotation()
