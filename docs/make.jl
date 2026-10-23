using Documenter
using TreeParzen

makedocs(
    source = @__DIR__,
    build = joinpath(@__DIR__, "..", "docs-build"),
    sitename = "TreeParzen.jl",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Getting started" => "usage.md",
        "Hyperparameter spaces" => "hyperparams.md",
        "Configuration" => "configuration.md",
        "Differences from Hyperopt" => "hyperopt-comparison.md",
        "MLJ integration" => "examples/simple_mlj_demo/simple_mlj_demo.md",
        "Development" => "development.md",
    ],
)