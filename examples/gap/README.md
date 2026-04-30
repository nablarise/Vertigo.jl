<!-- Copyright (c) 2026 Nablarise. All rights reserved. -->
<!-- Author: Guillaume Marques <guillaume@nablarise.com> -->
<!-- SPDX-License-Identifier: MIT -->

# GAP benchmark

Branch-cut-price benchmark on the **Generalized Assignment Problem**.

## Instance source

Instances are taken from **GAPLIB**, the GAP instance library curated
by V. Cacchiani, M. Iori, A. Locatelli, and S. Martello at the
University of Bologna:

  http://astarte.csr.unibo.it/gapdata/gapinstances.html

GAPLIB serves each instance as a small JSON document with the schema
`{name, numcli, numserv, cost::m×n, req::m×n, cap::m}`.

## Layout

| Path                         | Description                                  |
|------------------------------|----------------------------------------------|
| `instances/`                 | Smoke set — small JSONs, committed.          |
| `instances.zip`              | Full set — all GAPLIB JSONs, committed.      |
| `instances_metadata.json`    | Per-instance `(n, m, opt)` map.              |
| `gap_model.jl`               | `GAPInstance` + JSON parser + DW builder.    |
| `runner.jl`                  | Bench loop, metric capture, writers.         |
| `run_quick.jl`               | Entry point — smoke set.                     |
| `run_full.jl`                | Entry point — full set (unzips on demand).   |
| `download_instances.jl`      | One-off — rebuild `instances.zip` from web.  |
| `results/`                   | Output dir (gitignored).                     |

## Running

From the repository root:

```
julia --project=examples examples/gap/run_quick.jl
julia --project=examples examples/gap/run_full.jl
```

`run_quick.jl` runs the seven smoke instances under a 500-node B&P cap
(~1–2 min total). `run_full.jl` unpacks `instances.zip` to a temp
directory and runs every instance under a 10000-node cap.

To rebuild the instance archive from GAPLIB (rarely needed):

```
julia --project=examples examples/gap/download_instances.jl
```

## Metrics

Each run produces `results/<UTC-timestamp>-<mode>.{csv,json}` with one
row per instance. Captured columns:

| Field           | Source                                        |
|-----------------|-----------------------------------------------|
| `instance`      | name from JSON                                |
| `n`, `m`        | tasks, machines                               |
| `status`        | `BPOutput.status`                             |
| `wall_time_s`   | `@elapsed` around `run_branch_and_price`      |
| `primal_bound`  | `BPOutput.incumbent.obj_value`                |
| `dual_bound`    | `BPOutput.best_dual_bound`                    |
| `gap_pct`       | derived                                       |
| `opt_known`     | `instances_metadata.json` (when known)        |
| `nb_nodes`      | `BPOutput.nodes_explored`                     |
| `nb_columns`    | **stub** — pending issue #60                  |
| `nb_rmp_iters`  | **stub** — pending issue #61                  |
| `root_lp_value` | **stub** — pending issue #62                  |

The three stub fields are written as empty cells in CSV / `null` in
JSON until the corresponding `BPOutput` accessors land.

`status` and `dual_bound` are also affected by issue #63: when a
small instance is solved to optimality with the full tree explored,
`status` is currently reported as `:node_limit` and `dual_bound` as
`-Inf` (CSV) / `null` (JSON). The bench captures the values exactly
as `BPOutput` returns them; the status will normalize once #63 is
fixed.

## Smoke set

Seven instances under `instances/`:

| Instance      | n×m  | Optimum |
|---------------|------|--------:|
| `toy`         | 6×3  | unknown |
| `trivial`     | 6×3  | unknown |
| `gap1_0`      | 15×5 | 261     |
| `gap1_4`      | 15×5 | 251     |
| `beas_15_5_0` | 15×5 | 223     |
| `gap2_0`      | 20×5 | 277     |
| `arica`       | 40×5 | 5011    |

To extend the smoke set, drop another `<name>.json` into `instances/`
(and add an entry in `instances_metadata.json` if its optimum is
known).
