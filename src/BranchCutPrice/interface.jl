# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Accessors that dispatch on both ColGenContext and ColGenLoggerContext,
# so BranchCutPrice works with either type.

bp_master_model(ctx::ColGen.ColGenContext) = ctx.master_model
bp_master_model(ctx::ColGen.ColGenLoggerContext) = ctx.inner.master_model

bp_ip_incumbent(ctx::ColGen.ColGenContext) = ctx.ip_incumbent
bp_ip_incumbent(ctx::ColGen.ColGenLoggerContext) = ctx.inner.ip_incumbent

bp_pool(ctx::ColGen.ColGenContext) = ctx.pool
bp_pool(ctx::ColGen.ColGenLoggerContext) = ctx.inner.pool

bp_decomp(ctx::ColGen.ColGenContext) = ctx.decomp
bp_decomp(ctx::ColGen.ColGenLoggerContext) = ctx.inner.decomp
