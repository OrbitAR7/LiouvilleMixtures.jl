using Documenter
using LiouvilleMixtures

makedocs(;
    sitename = "LiouvilleMixtures.jl",
    modules = [LiouvilleMixtures],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", edit_link = nothing),
    pages = ["Home" => "index.md", "Derivations" => "derivations.md", "API" => "api.md"],
    remotes = nothing,   # no public remote yet; set when the repository is published
    checkdocs = :exports,
)
