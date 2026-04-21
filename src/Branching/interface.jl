# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Accessors that dispatch on both ColGenWorkspace and ColGenLoggerWorkspace,
# so BranchCutPrice works with either type.

bp_master_model(ws::ColGen.ColGenWorkspace) = master_model(ws.decomp)
bp_master_model(ws::ColGen.ColGenLoggerWorkspace) = master_model(ws.inner.decomp)

bp_ip_incumbent(ws::ColGen.ColGenWorkspace) = ws.ip_incumbent
bp_ip_incumbent(ws::ColGen.ColGenLoggerWorkspace) = ws.inner.ip_incumbent

bp_set_ip_incumbent!(ws::ColGen.ColGenWorkspace, val) =
    ws.ip_incumbent = val
bp_set_ip_incumbent!(ws::ColGen.ColGenLoggerWorkspace, val) =
    ws.inner.ip_incumbent = val

bp_pool(ws::ColGen.ColGenWorkspace) = ws.pool
bp_pool(ws::ColGen.ColGenLoggerWorkspace) = ws.inner.pool

bp_decomp(ws::ColGen.ColGenWorkspace) = ws.decomp
bp_decomp(ws::ColGen.ColGenLoggerWorkspace) = ws.inner.decomp

bp_branching_constraints(ws::ColGen.ColGenWorkspace) =
    ws.branching_constraints
bp_branching_constraints(ws::ColGen.ColGenLoggerWorkspace) =
    ws.inner.branching_constraints

bp_robust_cuts(ws::ColGen.ColGenWorkspace) = ws.robust_cuts
bp_robust_cuts(ws::ColGen.ColGenLoggerWorkspace) =
    ws.inner.robust_cuts

bp_ip_primal_bound(ws::ColGen.ColGenWorkspace) = ws.ip_primal_bound
bp_ip_primal_bound(ws::ColGen.ColGenLoggerWorkspace) =
    ws.inner.ip_primal_bound

bp_set_ip_primal_bound!(ws::ColGen.ColGenWorkspace, val) =
    ws.ip_primal_bound = val
bp_set_ip_primal_bound!(ws::ColGen.ColGenLoggerWorkspace, val) =
    ws.inner.ip_primal_bound = val
