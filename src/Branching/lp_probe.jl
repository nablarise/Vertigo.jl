# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# ────────────────────────────────────────────────────────────────────────────────────────
# LP-ONLY PROBING (cheap first phase)
# ────────────────────────────────────────────────────────────────────────────────────────

"""
    solve_master_lp_only!(backend) -> (obj_value, is_infeasible)

Solve the master LP and return the objective value and infeasibility flag.

Returns `(nothing, true)` if the LP is infeasible or infeasible/unbounded,
otherwise returns `(MOI.ObjectiveValue(backend), false)`.
"""
function solve_master_lp_only!(backend)
    MOI.optimize!(backend)
    status = MOI.get(backend, MOI.TerminationStatus())
    if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
        return (nothing, true)
    end
    obj = MOI.get(backend, MOI.ObjectiveValue())
    return (obj, false)
end

"""
    _run_one_lp_direction(space, candidate, set) -> SBProbeResult

Add a branching constraint for `candidate` with `set`, solve the master LP only,
remove the constraint, and return an `SBProbeResult`.

The constraint is always removed in a `finally` block.
"""
function _run_one_lp_direction(space, candidate, set)
    ctx = space.ctx
    backend = space.backend
    decomp = bp_decomp(ctx)
    pool = bp_pool(ctx)

    terms = build_branching_terms(decomp, pool, candidate.orig_var)
    ci = add_branching_constraint!(
        backend, ctx, terms, set, candidate.orig_var
    )

    try
        obj, is_inf = solve_master_lp_only!(backend)
        return SBProbeResult(obj, obj, is_inf)
    finally
        remove_branching_constraint!(backend, ctx, ci)
    end
end

"""
    probe_candidate(bctx, ::LPProbePhase, space, candidate, parent_lp)

LP probe: solve master LP only (no CG) in both directions.
Uses same state capture/restore as CG probes.
"""
function probe_candidate(
    bctx::BranchingContext, phase::LPProbePhase, space,
    candidate::BranchingCandidate, parent_lp::Float64
)
    snapshot = _capture_probe_state(space.ctx, space)
    try
        before_probe(bctx, phase, candidate, :left)
        left = _run_one_lp_direction(
            space, candidate,
            MOI.LessThan(candidate.floor_val)
        )
        after_probe(bctx, phase, candidate, :left, left)
        # Restore state so the right probe starts clean.
        _restore_probe_state!(space.ctx, space, snapshot)
        before_probe(bctx, phase, candidate, :right)
        right = _run_one_lp_direction(
            space, candidate,
            MOI.GreaterThan(candidate.ceil_val)
        )
        after_probe(bctx, phase, candidate, :right, right)
        return SBCandidateResult(
            candidate, parent_lp, left, right
        )
    finally
        _restore_probe_state!(space.ctx, space, snapshot)
    end
end
