# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Test

push!(LOAD_PATH, joinpath(@__DIR__, "VertigoUnitTests"))
push!(LOAD_PATH, joinpath(@__DIR__, "VertigoE2eTests"))

using VertigoUnitTests
using VertigoE2eTests

VertigoUnitTests.run()
VertigoE2eTests.run()
