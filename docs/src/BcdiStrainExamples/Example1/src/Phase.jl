using BcdiStrain
using BcdiSimulate
using Plots
using Statistics
using FFTW

function saveAn(state, a)
    p1 = heatmap(fftshift(Array(reshape(state.rho,100,100,100))[1,:,:]))
    p2 = heatmap(fftshift(Array(reshape(state.ux,100,100,100))[1,:,:]))
    p3 = heatmap(fftshift(Array(reshape(state.uy,100,100,100))[1,:,:]))
    p4 = heatmap(fftshift(Array(reshape(state.uz,100,100,100))[1,:,:]))
    p5 = heatmap(fftshift(Array(reshape(abs.(state.traditionals[1].realSpace),100,100,100)[1,:,:])))
    p6 = heatmap(fftshift(Array(reshape(angle.(state.traditionals[1].realSpace),100,100,100)[1,:,:])))
    frame(a, plot(p1,p2,p5,p3,p4,p6,layout=6,size=(900,400)))
end

function simulateDiffraction()
    hRanges = [
        range(0.25-0.03,0.25-0.03+100*0.06/101,100),
        range(0.25-0.03,0.25-0.03+100*0.06/101,100),
        range(-0.25-0.03,-0.25-0.03+100*0.06/101,100)
    ]
    kRanges = [
        range(0.25-0.03,0.25-0.03+100*0.06/101,100),
        range(-0.25-0.03,-0.25-0.03+100*0.06/101,100),
        range(-0.25-0.03,-0.25-0.03+100*0.06/101,100)
    ]
    lRanges = [
        range(-0.25-0.03,-0.25-0.03+100*0.06/101,100),
        range(0.25-0.03,0.25-0.03+100*0.06/101,100),
        range(-0.25-0.03,-0.25-0.03+100*0.06/101,100)
    ]
    rotations = [
        [1. 0 0 ; 0 1 0 ; 0 0 1],
        [1. 0 0 ; 0 1 0 ; 0 0 1],
        [1. 0 0 ; 0 1 0 ; 0 0 1]
    ]

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
    state = BcdiStrain.State(intens, gVecs, recSupport)
    er = BcdiStrain.ER()
    hio = BcdiStrain.HIO(0.01)
    shrink = BcdiStrain.Shrink(0.1, 1.0, state)
    center = BcdiStrain.Center(state)
    mount = BcdiStrain.Mount(0.5, state, recPrimLatt)

    a = Animation()
    # We could run the commands this way, but we want to plot in the middle
    # (mount * center * er^20)^200 * 
    # (mount * center * (shrink * hio)^80)^20 * state

    for i in 1:5
        for j in 1:320
            println(i," ",j)
            hio * state
            saveAn(state, a)
            shrink * state
        end
        center * state
        saveAn(state, a)
        mount * state
        saveAn(state, a)
    end
    for i in 1:200
        println(i)
        er * state
        center * state
        saveAn(state, a)
        mount * state
    end
    saveAn(state, a)
    mov(a, "../results/recon.webm", fps=200)
end

intens, recSupport, GMaxs, recPrimLatt, rotations = simulateDiffraction()
phase(intens, recSupport, GMaxs, recPrimLatt, rotations)
