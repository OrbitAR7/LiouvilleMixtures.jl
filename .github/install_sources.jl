# Add the git dependencies listed under [sources] in a Project.toml, at their
# pinned revisions. Julia >= 1.11 reads [sources] itself. Julia 1.10 ignores
# it, so CI on the LTS runs this script first.
#
#   julia --project=<env> .github/install_sources.jl <Project.toml with [sources]>
using Pkg, TOML

project = TOML.parsefile(get(ARGS, 1, "Project.toml"))
specs = [PackageSpec(name=name, url=s["url"], rev=s["rev"])
         for (name, s) in get(project, "sources", Dict()) if haskey(s, "url")]
isempty(specs) || Pkg.add(specs)
Pkg.instantiate()
