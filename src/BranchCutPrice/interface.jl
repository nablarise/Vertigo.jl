# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Accessors that dispatch on both ColGenContext and ColGenLoggerContext,
# so BranchCutPrice works with either type.

bp_master_model(ctx::ColGen.ColGenContext) = master_model(ctx.decomp)
bp_master_model(ctx::ColGen.ColGenLoggerContext) = master_model(ctx.inner.decomp)

bp_ip_incumbent(ctx::ColGen.ColGenContext) = ctx.ip_incumbent
bp_ip_incumbent(ctx::ColGen.ColGenLoggerContext) = ctx.inner.ip_incumbent

bp_pool(ctx::ColGen.ColGenContext) = ctx.pool
bp_pool(ctx::ColGen.ColGenLoggerContext) = ctx.inner.pool

bp_decomp(ctx::ColGen.ColGenContext) = ctx.decomp
bp_decomp(ctx::ColGen.ColGenLoggerContext) = ctx.inner.decomp

bp_branching_constraints(ctx::ColGen.ColGenContext) =
    ctx.branching_constraints
bp_branching_constraints(ctx::ColGen.ColGenLoggerContext) =
    ctx.inner.branching_constraints

bp_robust_cuts(ctx::ColGen.ColGenContext) = ctx.robust_cuts
bp_robust_cuts(ctx::ColGen.ColGenLoggerContext) =
    ctx.inner.robust_cuts
