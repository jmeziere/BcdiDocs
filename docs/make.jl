using Documenter, DocumenterCitations, BcdiTrad, BcdiStrain, BcdiMeso

inEntry = false
newEntry = true
entryName = ""
entry = String[]
entryDict =  Dict{String,Vector{String}}()

inBibFiles = [
    "docs/src/BcdiCore.jl/docs/src/refs.bib"
    "docs/src/BcdiTrad.jl/docs/src/refs.bib"
    "docs/src/BcdiStrain.jl/docs/src/refs.bib"
    "docs/src/BcdiMeso.jl/docs/src/refs.bib"
    "docs/src/BcdiSimulate.jl/docs/src/refs.bib"
]
outBibFile = "docs/src/refs.bib"
for filename in inBibFiles
    open(filename,"r") do fin
        while !eof(fin)
            global inEntry
            global newEntry
            global entryName
            global entry
            line = readline(fin)
            if inEntry == false && occursin("@", line)
                inEntry = true
                startInd = findfirst("{",line)[1]+1
                endInd = findfirst(",",line)[1]-1
                entryName = line[startInd:endInd]
                if entryName in keys(entryDict)
                    newEntry = false
                else
                    newEntry = true
                end
                entry = String[line]
            elseif inEntry == true && newEntry
                if strip(line, [' ','\n']) == "}"
                   push!(entry, line)
                   entryDict[entryName] = entry
                   inEntry = false
                else
                   push!(entry, line)
                end
            end
        end
    end
end

open(outBibFile,"w") do fout
    for k in keys(entryDict)
        for line in entryDict[k]
            println(fout, line)
        end
        println(fout, "")
    end
end

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename="BcdiDocs",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Julia BCDI"=>"index.md",
        "BcdiCore"=>[
            "About/Installation"=>"BcdiCore.jl/docs/src/index.md",
            "Overview"=>"BcdiCore.jl/docs/src/use/overview.md",
            "Atomic Models"=>"BcdiCore.jl/docs/src/use/atomic.md",
            "Mesoscale Models"=>"BcdiCore.jl/docs/src/use/meso.md",
            "Traditional Models"=>"BcdiCore.jl/docs/src/use/trad.md",
            "Multiscale Models"=>"BcdiCore.jl/docs/src/use/multi.md"
        ],
        "BcdiTrad"=>[
            "About/Installation"=>"BcdiTrad.jl/docs/src/index.md",
            "Usage"=>"BcdiTrad.jl/docs/src/use.md",
            "Examples"=>"BcdiTradExamples/examples.md"
        ],
        "BcdiStrain"=>[
            "About/Installation"=>"BcdiStrain.jl/docs/src/index.md",
            "Usage"=>"BcdiStrain.jl/docs/src/use.md",
            "Examples"=>"BcdiStrainExamples/examples.md"
        ],
        "BcdiMeso"=>[
            "About/Installation"=>"BcdiMeso.jl/docs/src/index.md",
            "Usage"=>"BcdiMeso.jl/docs/src/use.md",
            "Examples"=>"BcdiMesoExamples/examples.md"
        ],
        "BcdiSimulate"=>[
            "About/Installation"=>"BcdiSimulate.jl/docs/src/index.md",
            "Atomic Simulation"=>"BcdiSimulate.jl/docs/src/usage/atom.md"
        ],
        "References"=>"refs.md"
    ],
    remotes = Dict(
        "src/BcdiCore.jl"=>Documenter.Remotes.GitHub("byu-cxi", "BcdiCore.jl.git"),
        "src/BcdiTrad.jl"=>Documenter.Remotes.GitHub("byu-cxi", "BcdiTrad.jl.git"),
        "src/BcdiStrain.jl"=>Documenter.Remotes.GitHub("byu-cxi", "BcdiStrain.jl.git"),
        "src/BcdiMeso.jl"=>Documenter.Remotes.GitHub("byu-cxi", "BcdiMeso.jl.git")
        "src/BcdiSimulate.jl"=>Documenter.Remotes.GitHub("byu-cxi", "BcdiSimulate.jl.git")
    ),
    plugins = [bib]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiDocs.git",
)
