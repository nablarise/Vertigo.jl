using Documenter
using Vertigo

makedocs(;
    sitename="Vertigo.jl",
    repo=Documenter.Remotes.GitHub("nablarise", "Vertigo.jl"),
    modules=[
        Vertigo,
        Vertigo.DML,
        Vertigo.Reformulation,
        Vertigo.ColGen,
        Vertigo.BranchCutPrice,
        Vertigo.Branching,
        Vertigo.TreeSearch,
        Vertigo.MathOptState,
    ],
    checkdocs=:none,
    format=Documenter.HTML(;
        edit_link="main", size_threshold_warn=200 * 2^10, size_threshold=400 * 2^10
    ),
    pages=[
        "Introduction" => "index.md",
        "Tutorials" => ["tutorials/gap.md", "tutorials/overriding-subproblems.md"],
        "User API" =>
            ["user-api/decomposition.md", "user-api/solving.md", "user-api/errors.md"],
        "Internal API" => [
            "internal-api/introduction.md",
            "internal-api/dml.md",
            "internal-api/master.md",
            "internal-api/subproblem.md",
            "internal-api/column-generation.md",
            "internal-api/branch-and-price.md",
        ],
        "API Reference" => [
            "reference/index.md",
            "reference/dml.md",
            "reference/reformulation.md",
            "reference/column-generation.md",
            "reference/branch-and-price.md",
            "reference/branching.md",
            "reference/tree-search.md",
            "reference/model-state.md",
        ],
        "FAQ" => "faq.md",
    ],
)

deploydocs(; repo="github.com/nablarise/Vertigo.jl.git", devbranch="main")
