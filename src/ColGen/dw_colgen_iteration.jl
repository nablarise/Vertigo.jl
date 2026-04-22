# Copyright (c) 2025 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# This file is intentionally empty.
# The column generation iteration logic lives in src/coluna.jl (the shared kernel).
# Dispatch implementations live in their respective specialized files:
#   - master_optimization.jl
#   - reduced_costs.jl
#   - pricing_optimization.jl
#   - column_insertion.jl
#   - dual_bounds.jl
#   - dw_stabilization.jl
#   - ip_management.jl
