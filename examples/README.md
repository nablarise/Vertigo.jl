<!-- Copyright (c) 2026 Nablarise. All rights reserved. -->
<!-- Author: Guillaume Marques <guillaume@nablarise.com> -->
<!-- SPDX-License-Identifier: MIT -->

# Vertigo benchmarks

Benchmark suites for tracking Vertigo's performance over time on
standard column-generation / branch-cut-price problem classes.

This folder is for **benchmarking**, not pedagogy: each example is a
runner that solves a fixed set of instances under a fixed Vertigo
configuration, captures metrics (time, nodes, gap, columns, …), and
writes them to CSV/JSON for diffing between commits.

## Layout

| Path        | Problem                          |
|-------------|----------------------------------|
| `gap/`      | Generalized Assignment Problem   |

## Running

Each suite has its own `run_quick.jl` (smoke set, ~minutes) and
`run_full.jl` (full set, ~hours). Activate the `examples/` project:

```
julia --project=examples examples/gap/run_quick.jl
julia --project=examples examples/gap/run_full.jl
```

Results land under `<suite>/results/` (gitignored).
