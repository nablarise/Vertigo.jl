# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: Proprietary

_eachindex(obj) = eachindex(obj)
_eachindex(obj::Matrix) = [(i, j) for i in axes(obj, 1), j in axes(obj, 2)]

JuMP.Containers.default_container(::Vector{Tuple{Int64,Int64}}) = JuMP.Containers.SparseAxisArray
JuMP.Containers.default_container(::Vector{<:Tuple}) = JuMP.Containers.SparseAxisArray
