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
            "Usage"=>"BcdiTrad.jl/docs/src/use.md"
        ],
        "BcdiStrain"=>[
            "About/Installation"=>"BcdiStrain.jl/docs/src/index.md",
            "Usage"=>"BcdiStrain.jl/docs/src/use.md"
        ],
        "BcdiMeso"=>[
            "About/Installation"=>"BcdiMeso.jl/docs/src/index.md",
            "Usage"=>"BcdiMeso.jl/docs/src/use.md"
        ]
    ],
    remotes = Dict(
        "BcdiCore.jl"=>GitHub("byu.cxi", "BcdiCore.jl"),
        "BcdiTrad.jl"=>GitHub("byu.cxi", "BcdiTrad.jl"),
        "BcdiStrain.jl"=>GitHub("byu.cxi", "BcdiStrain.jl"),
        "BcdiMeso.jl"=>GitHub("byu.cxi", "BcdiMeso.jl")
    )
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiDocs.git",
)
