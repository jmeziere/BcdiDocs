using Documenter, BcdiTrad, BcdiStrain, BcdiMeso

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
            "Usage"=>"BcdiStrain.jl/docs/src/use.md"
            "Examples"=>"BcdiStrainExamples/examples.md"
        ],
        "BcdiMeso"=>[
            "About/Installation"=>"BcdiMeso.jl/docs/src/index.md",
            "Usage"=>"BcdiMeso.jl/docs/src/use.md"
            "Examples"=>"BcdiMesoExamples/examples.md"
        ]
    ],
    remotes = Dict(
        "src/BcdiCore.jl"=>Documenter.Remotes.GitHub("byu-cxi", "BcdiCore.jl.git"),
        "src/BcdiTrad.jl"=>Documenter.Remotes.GitHub("byu-cxi", "BcdiTrad.jl.git"),
        "src/BcdiStrain.jl"=>Documenter.Remotes.GitHub("byu-cxi", "BcdiStrain.jl.git"),
        "src/BcdiMeso.jl"=>Documenter.Remotes.GitHub("byu-cxi", "BcdiMeso.jl.git")
    )
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiDocs.git",
)
