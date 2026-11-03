# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

using Test

push!(LOAD_PATH, joinpath(@__DIR__, "VertigoUnitTests"))
push!(LOAD_PATH, joinpath(@__DIR__, "VertigoE2eTests"))

using VertigoUnitTests
using VertigoE2eTests

# One outer testset so that unit failures do not keep the e2e tests from running.
@testset "Vertigo" begin
    VertigoUnitTests.run()
    VertigoE2eTests.run()
end
