# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Accessors that dispatch on both ColGenWorkspace and ColGenLoggerContext,
# so BranchCutPrice works with either type.

bp_master_model(ctx::ColGen.ColGenWorkspace) = master_model(ctx.decomp)
bp_master_model(ctx::ColGen.ColGenLoggerContext) = master_model(ctx.inner.decomp)

bp_ip_incumbent(ctx::ColGen.ColGenWorkspace) = ctx.ip_incumbent
bp_ip_incumbent(ctx::ColGen.ColGenLoggerContext) = ctx.inner.ip_incumbent

bp_set_ip_incumbent!(ctx::ColGen.ColGenWorkspace, val) =
    ctx.ip_incumbent = val
bp_set_ip_incumbent!(ctx::ColGen.ColGenLoggerContext, val) =
    ctx.inner.ip_incumbent = val

bp_pool(ctx::ColGen.ColGenWorkspace) = ctx.pool
bp_pool(ctx::ColGen.ColGenLoggerContext) = ctx.inner.pool

bp_decomp(ctx::ColGen.ColGenWorkspace) = ctx.decomp
bp_decomp(ctx::ColGen.ColGenLoggerContext) = ctx.inner.decomp

bp_branching_constraints(ctx::ColGen.ColGenWorkspace) =
    ctx.branching_constraints
bp_branching_constraints(ctx::ColGen.ColGenLoggerContext) =
    ctx.inner.branching_constraints

bp_robust_cuts(ctx::ColGen.ColGenWorkspace) = ctx.robust_cuts
bp_robust_cuts(ctx::ColGen.ColGenLoggerContext) =
    ctx.inner.robust_cuts

bp_ip_primal_bound(ctx::ColGen.ColGenWorkspace) = ctx.ip_primal_bound
bp_ip_primal_bound(ctx::ColGen.ColGenLoggerContext) =
    ctx.inner.ip_primal_bound

bp_set_ip_primal_bound!(ctx::ColGen.ColGenWorkspace, val) =
    ctx.ip_primal_bound = val
bp_set_ip_primal_bound!(ctx::ColGen.ColGenLoggerContext, val) =
    ctx.inner.ip_primal_bound = val
