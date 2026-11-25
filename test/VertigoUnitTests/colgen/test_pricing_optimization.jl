using Vertigo.ColGen: is_subproblem_optional

function test_colgen_pricing_optimization_optional_subproblem()
    for (name, conv_lb, expected) in [
        ("lower multiplicity 0 is optional", 0.0, true),
        ("lower multiplicity 1 is mandatory", 1.0, false),
        ("lower multiplicity 2 is mandatory", 2.0, false),
    ]
        @testset "[pricing_optimization] $name" begin
            sp_id = PricingSubproblemId(1)
            builder = DWReformulationBuilder{Int}(minimize=true)
            add_subproblem!(builder, sp_id, 0.0, conv_lb, 2.0)
            ws = ColGenWorkspace(build(builder), ColGenConfig())

            @test is_subproblem_optional(ws, sp_id) == expected
        end
    end
end
