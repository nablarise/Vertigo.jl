# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

include("runner.jl")

function main()
    instances_dir = joinpath(@__DIR__, "instances")
    paths = sort(filter(p -> endswith(p, ".json"),
        readdir(instances_dir; join=true)))

    isempty(paths) && error(
        "no instances under $(instances_dir); run download_instances.jl first"
    )

    cfg = BenchConfig(
        500,    # node_limit
        0.5,    # smoothing_alpha
        10.0    # rmp_time_limit (per-node IP heuristic)
    )

    run_bench(paths, cfg, "quick")
end

main()
