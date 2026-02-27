# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

struct ProjectedIpPrimalSol end

function check_primal_ip_feasibility!(
    ::MasterPrimalSolution, ::ColGenContext, ::Union{Phase0,Phase1,Phase2}
)
    return ProjectedIpPrimalSol(), false
end

function update_inc_primal_sol!(::ColGenContext, ::Nothing, ::ProjectedIpPrimalSol)
    return nothing
end
