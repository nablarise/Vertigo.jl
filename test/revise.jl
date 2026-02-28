# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

# Interactive dev loop: run this file from the REPL with include("test/revise.jl")
# Revise will automatically pick up changes to ColumnGeneration source files.

using Revise
using Vertigo

push!(LOAD_PATH, joinpath(@__DIR__, "VertigoUnitTests"))
push!(LOAD_PATH, joinpath(@__DIR__, "VertigoE2eTests"))

using VertigoUnitTests
using VertigoE2eTests

VertigoUnitTests.run()
VertigoE2eTests.run()
