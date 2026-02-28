# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

setup_stabilization!(::ColGenContext, master) = NoStabilization()

function update_stabilization_after_master_optim!(
    ::NoStabilization, phase, ::MasterDualSolution
)
    return false
end

get_stab_dual_sol(::NoStabilization, phase, dual_sol::MasterDualSolution) = dual_sol

function update_stabilization_after_pricing_optim!(
    ::NoStabilization, ::ColGenContext, _, _, _, _
)
    return nothing
end

check_misprice(::NoStabilization, _, _) = false

update_stabilization_after_iter!(::NoStabilization, ::MasterDualSolution) = nothing
