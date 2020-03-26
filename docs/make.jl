using Documenter, NoiseReduction

makedocs(;
    modules=[NoiseReduction],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/ssfrr/NoiseReduction.jl/blob/{commit}{path}#L{line}",
    sitename="NoiseReduction.jl",
    authors="Spencer Russell",
    assets=String[],
)

deploydocs(;
    repo="github.com/ssfrr/NoiseReduction.jl",
)
