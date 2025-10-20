push!(LOAD_PATH, "../src/")
using Documenter
using .ICS

DocMeta.setdocmeta!(Robustbase, :DocTestSetup, :(using Robustbase); recursive=true)

makedocs(sitename = "ICS",
    format = Documenter.HTML(),
    modules = [ICS],
    warnonly = :missing_docs
    ## warnonly = [:missing_docs, :docs_block]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/valentint/ICS.jl.git", devbranch = "main"
)
