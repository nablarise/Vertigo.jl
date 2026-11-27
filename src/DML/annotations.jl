# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

struct SubproblemAnnotation
    id::Any
end

struct MasterAnnotation
end

# A subproblem variable whose value is the column's coefficient in the
# convexity rows of its subproblem.
struct SetupAnnotation
    id::Any
end

dantzig_wolfe_subproblem(id) = SubproblemAnnotation(id)
dantzig_wolfe_master() = MasterAnnotation()
dantzig_wolfe_setup(id) = SetupAnnotation(id)
