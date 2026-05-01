# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using ZipFile

include("runner.jl")

function unzip_to(zip_path::String, dest::String)
    mkpath(dest)
    zf = ZipFile.Reader(zip_path)
    try
        for f in zf.files
            endswith(f.name, ".json") || continue
            out = joinpath(dest, basename(f.name))
            open(out, "w") do io
                write(io, read(f))
            end
        end
    finally
        close(zf)
    end
end

function main()
    zip_path = joinpath(@__DIR__, "instances.zip")
    isfile(zip_path) || error("missing $(zip_path)")

    workdir = mktempdir()
    unzip_to(zip_path, workdir)
    paths = sort(filter(p -> endswith(p, ".json"),
        readdir(workdir; join=true)))

    cfg = BenchConfig(
        10_000, # node_limit
        0.5,    # smoothing_alpha
        60.0    # rmp_time_limit
    )

    run_bench(paths, cfg, "full")
end

main()
