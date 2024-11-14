using BcdiSimulate
using BcdiAtomic

function simulateDiffraction()
    hRanges = [
        range(-0.4,-0.1,50),
        range(0.1,0.4,50),
        range(0.1,0.4,50)
    ]
    kRanges = [
        range(0.1,0.4,50),
        range(-0.4,-0.1,50),
        range(0.1,0.4,50)
    ]
    lRanges = [
        range(0.1,0.4,50),
        range(0.1,0.4,50),
        range(-0.4,-0.1,50)
    ]

    x = parse.(Float64, readlines("../data/xvals.csv"))
    y = parse.(Float64, readlines("../data/yvals.csv"))
    z = parse.(Float64, readlines("../data/zvals.csv"))

    lammpsOptions = [
        "-screen","none",
        "-log","none"#,
#        "-sf","gpu",
#        "-pk","gpu","1"
    ]

    numPhotons = [1e12,1e12,1e12]

    BcdiSimulate.relaxCrystal(x, y, z, lammpsOptions, "../data/Au_Zhou04.eam.alloy Au")
    intens, recSupport, Gs, boxSize = BcdiSimulate.atomSimulateDiffraction(x, y, z, hRanges, kRanges, lRanges, numPhotons)

    recipLatt = zeros(3,3)
    for i in 1:length(Gs)
        recipLatt[i,:] .= Gs[i]
    end
    xRange = 0.0:1.0:99.0
    yRange = 0.0:1.0:99.0
    zRange = 0.0:1.0:99.0
    BcdiSimulate.generateDisplacement(x, y, z, recipLatt, xRange, yRange, zRange)

    exit()
    return round(Int32, 0.9*length(x)), round(Int32, 1.1*length(x)), intens, recSupport, Gs, boxSize
end

function Phase(minAtom, maxAtom, intens, recSupport, Gs, boxSize)
    x = Float64[]
    y = Float64[]
    z = Float64[]
    numBox = ceil(Int64, (minAtom / 4)^(1/3))
    for i in 1:numBox
        for j in 1:numBox
            for k in 1:numBox
                 append!(x, [4*i, 4*i+2, 4*i+2, 4*i])
                 append!(y, [4*j, 4*j+2, 4*j, 4*j+2])
                 append!(z, [4*k, 4*k, 4*k+2, 4*k+2])
            end
        end
    end

    state = BcdiAtomic.State(
        minAtom, maxAtom, 
        "../data/Au_Zhou04.eam.alloy Au", 
        x, y, z, intens, recSupport, Gs, boxSize
    )
    BcdiAtomic.prammol!(state)
end

minAtom, maxAtom, intens, recSupport, Gs, boxSize = simulateDiffraction()
Phase(minAtom, maxAtom, intens, recSupport, Gs, boxSize)
