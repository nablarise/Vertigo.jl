# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

using Test

push!(LOAD_PATH, joinpath(@__DIR__, "ColumnGenerationUnitTests"))

using ColumnGenerationUnitTests

ColumnGenerationUnitTests.run()
