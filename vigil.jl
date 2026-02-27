# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Add test modules to LOAD_PATH
for submodule in filter(item -> isdir(joinpath(@__DIR__, "test", item)), readdir(joinpath(@__DIR__, "test")))
    push!(LOAD_PATH, joinpath(@__DIR__, "test", submodule))
end

using Vigil
using ColumnGeneration
using ColumnGenerationUnitTests
using ColumnGenerationE2eTests

# Parse command line arguments
output_file = length(ARGS) >= 1 ? abspath(ARGS[1]) : nothing
format = isnothing(output_file) ? Vigil.Plain : Vigil.JSONL

if isnothing(output_file)
    println("Starting test watcher in interactive mode...")
    println("Press Ctrl+C to stop.")
else
    println("Starting test watcher with output to: $output_file")
    println("Press Ctrl+C to stop.")
end

watch_and_run(
    [ColumnGenerationUnitTests, ColumnGenerationE2eTests],
    [ColumnGeneration];
    output_file=output_file,
    format=format
)
