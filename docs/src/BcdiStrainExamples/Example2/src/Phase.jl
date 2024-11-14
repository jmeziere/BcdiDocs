using BcdiStrain
using BcdiSimulate
using Plots
using Statistics
using FFTW

function saveAn(state, a)
    realSpace = zeros(ComplexF64,100,100,100)
    realSpace[Array(state.keepInd)] .= Array(state.traditionals[1].realSpace)
    p1 = heatmap(Array(state.rho)[50,:,:])
    p2 = heatmap(Array(state.ux)[50,:,:])
    p3 = heatmap(Array(state.uy)[50,:,:])
    p4 = heatmap(Array(state.uz)[50,:,:])
    p5 = heatmap(abs.(realSpace)[50,:,:])
    p6 = heatmap(angle.(realSpace)[50,:,:])
    frame(a, plot(p1,p2,p5,p3,p4,p6,layout=6,size=(900,400)))
end

function simulateDiffraction()
    hRanges = [
        range(sqrt(3*0.25^2)-0.03,sqrt(3*0.25^2)-0.03+100*0.06/101,100),
        range(sqrt(3*0.25^2)-0.03,sqrt(3*0.25^2)-0.03+100*0.06/101,100),
        range(sqrt(3*0.25^2)-0.03,sqrt(3*0.25^2)-0.03+100*0.06/101,100)
    ]
    kRanges = [
        range(-0.03,-0.03+100*0.06/101,100),
        range(-0.03,-0.03+100*0.06/101,100),
        range(-0.03,-0.03+100*0.06/101,100)
    ]
    lRanges = [
        range(-0.03,-0.03+100*0.06/101,100),
        range(-0.03,-0.03+100*0.06/101,100),
        range(-0.03,-0.03+100*0.06/101,100)
    ]
    rotations = BcdiSimulate.getRotations(
        [[1,0,0],[1,0,0],[1,0,0]],
        [[0.25,0.25,-0.25],[0.25,-0.25,0.25],[-0.25,0.25,0.25]]
    )

    x = parse.(Float64, readlines("../data/xvals.csv"))
    x .-= mean(x)
    x = Float32.(x)
    y = parse.(Float64, readlines("../data/yvals.csv"))
    y .-= mean(y)
    y = Float32.(y)
    z = parse.(Float64, readlines("../data/zvals.csv"))
    z .-= mean(z)
    z = Float32.(z)

    lammpsOptions = [
        "-screen","none",
        "-log","none",
        "-sf","gpu",
        "-pk","gpu","1","neigh","no"
    ]

    numPhotons = [1e12,1e12,1e12]

#    BcdiSimulate.relaxCrystal(x, y, z, lammpsOptions, "../data/Au_Zhou04.eam.alloy Au")
    x = Float64.(x)
    y = Float64.(y)
    z = Float64.(z)
    intens, recSupport, GCens, GMaxs, boxSize = BcdiSimulate.atomSimulateDiffraction(x, y, z, hRanges, kRanges, lRanges, rotations, numPhotons)

    recPrimLatt = zeros(3,3)
    for i in 1:3
        GMaxs[i] .*= boxSize
        recPrimLatt[i,:] .= GMaxs[i]
    end
    return intens, recSupport, GMaxs, recPrimLatt, rotations
end

function phase(intens, recSupport, gVecs, recPrimLatt, rotations)
    state = BcdiStrain.State(intens, gVecs, recSupport, rotations=rotations)
    eropt = BcdiStrain.EROpt(state)
    er = BcdiStrain.ER()
    hioopt = BcdiStrain.HIOOpt(2.0, state)
    shrink = BcdiStrain.Shrink(0.1, 1.0, state)
    center = BcdiStrain.Center(state)
    mount = BcdiStrain.Mount(0.5, state, recPrimLatt)

    a = Animation()
    # We could run the commands this way, but we want to plot in the middle
    # (mount * center * er^20)^200 * 
    # (mount * center * (shrink * hio)^80)^20 * state

    for i in 1:10
        for j in 1:160
            println(i," ",j)
            hioopt * state
            saveAn(state, a)
            shrink * state
        end
        eropt^2 * state
        center * state
        mount * state
        saveAn(state, a)
    end
    for i in 1:100
        println(i)
        eropt * state
        center * state
        mount * state
        saveAn(state, a)
    end
    for i in 1:100
        println(i)
        er * state
        center * state
        mount * state
        saveAn(state, a)
    end
    mov(a, "../results/recon.webm", fps=200)
end

intens, recSupport, GMaxs, recPrimLatt, rotations = simulateDiffraction()
phase(intens, recSupport, GMaxs, recPrimLatt, rotations)
