# Copyright (c) 2026 Nablarise. All rights reserved.
# Author: Guillaume Marques <guillaume@nablarise.com>
# SPDX-License-Identifier: MIT

# One-off tool: scrape the GAPLIB instance index, download every JSON
# instance, pack them into examples/gap/instances.zip, and write a
# sidecar examples/gap/instances_metadata.json with (n, m, opt) per
# instance. Re-run only when GAPLIB updates.
#
# Source: GAPLIB — http://astarte.csr.unibo.it/gapdata/gapinstances.html

using HTTP
using JSON
using Printf
using ZipFile

const BASE_URL = "http://astarte.csr.unibo.it/gapdata/"
const INDEX_URL = BASE_URL * "gapinstances.html"
const OUTPUT_ZIP = joinpath(@__DIR__, "instances.zip")
const OUTPUT_META = joinpath(@__DIR__, "instances_metadata.json")

# Each table row in the index has the shape:
#   <td>id</td> <td>n</td> <td>m</td> ... <td><a href="...txt">...</a></td>
#   <td><a href="...json">...</a></td> <td>notes</td>
# Use a regex to extract every (id, n, m, opt, json_href) tuple.
const ROW_PATTERN = r"<tr>\s*<td>\s*([A-Za-z0-9_]+)\s*</td>\s*<td>\s*(\d+)\s*</td>\s*<td>\s*(\d+)\s*</td>\s*<td>([^<]*)</td>\s*<td>([^<]*)</td>\s*<td>([^<]*)</td>.*?<a\s+href=\"\s*([^\"]*\.json)\s*\""s

struct IndexEntry
    id::String
    n::Int
    m::Int
    opt::Union{Float64,Nothing}
    href::String
end

function parse_opt(field::AbstractString)
    s = strip(replace(field, r"&[^;]+;" => ""))
    isempty(s) || s == "-" && return nothing
    v = tryparse(Float64, s)
    return v
end

function fetch_index()
    resp = HTTP.get(INDEX_URL; require_ssl_verification=false)
    resp.status == 200 || error("GET $(INDEX_URL) returned $(resp.status)")
    return String(resp.body)
end

function parse_index(html::AbstractString)::Vector{IndexEntry}
    entries = IndexEntry[]
    for m in eachmatch(ROW_PATTERN, html)
        id = strip(m.captures[1])
        n = parse(Int, m.captures[2])
        mm = parse(Int, m.captures[3])
        opt = parse_opt(m.captures[5])
        href = String(strip(m.captures[7]))
        # GAPLIB sometimes uses backslashes in hrefs; normalize
        href = replace(href, "\\" => "/")
        push!(entries, IndexEntry(id, n, mm, opt, href))
    end
    return entries
end

function fetch_instance(href::AbstractString)
    url = startswith(href, "http") ? href : BASE_URL * href
    resp = HTTP.get(url; require_ssl_verification=false)
    resp.status == 200 || error("GET $(url) returned $(resp.status)")
    return resp.body
end

function main()
    println("Fetching GAPLIB index ...")
    html = fetch_index()
    entries = parse_index(html)
    println("found $(length(entries)) instances")

    isempty(entries) && error("no instances parsed; check ROW_PATTERN")

    metadata = Dict{String,Any}()
    archive = ZipFile.Writer(OUTPUT_ZIP)

    try
        for (i, e) in enumerate(entries)
            @printf("[%4d/%d] %-30s (%dx%d) ... ",
                i, length(entries), e.id, e.n, e.m)
            flush(stdout)
            try
                bytes = fetch_instance(e.href)
                w = ZipFile.addfile(archive, "$(e.id).json")
                write(w, bytes)
                metadata[e.id] = Dict(
                    "n" => e.n, "m" => e.m, "opt" => e.opt
                )
                println("ok ($(length(bytes)) B)")
            catch err
                println("FAILED ($(typeof(err)))")
            end
        end
    finally
        close(archive)
    end

    open(OUTPUT_META, "w") do io
        JSON.print(io, metadata, 2)
    end

    println()
    println("wrote $(OUTPUT_ZIP)")
    println("wrote $(OUTPUT_META)")
end

main()
