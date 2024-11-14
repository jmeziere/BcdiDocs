using BcdiTrad
using BcdiSimulate
using Plots
using FFTW
using LinearAlgebra
using Statistics

function saveAn(state, a)
    p1 = heatmap(fftshift(Array(abs.(state.realSpace)))[50,:,:])
    p2 = heatmap(fftshift(Array(angle.(state.realSpace)))[50,:,:])
    frame(a, plot(p1,p2,layout=2,size=(600,200)))
end

function simulateDiffraction()
    hRanges = [
        range(sqrt(3*0.25^2)-0.03,sqrt(3*0.25^2)-0.03+100*0.06/101,100)
    ]
    kRanges = [
        range(-0.03,-0.03+100*0.06/101,100)
    ]
    lRanges = [
        range(-0.03,-0.03+100*0.06/101,100)
    ]
    rotations = BcdiSimulate.getRotations([[1,0,0]],[[0.25,0.25,-0.25]])

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

    BcdiSimulate.relaxCrystal(x, y, z, lammpsOptions, "../data/Au_Zhou04.eam.alloy Au")
    x = Float64.(x)
    y = Float64.(y)
    z = Float64.(z)
    intens, recSupport, GCens, GMaxs, boxSize = BcdiSimulate.atomSimulateDiffraction(x, y, z, hRanges, kRanges, lRanges, rotations, numPhotons)
    return intens[1], recSupport[1]
end

function phase(intensities, recSupport)
    state = BcdiTrad.State(intensities, recSupport)
    er = BcdiTrad.ER()
    hio = BcdiTrad.HIO(0.9)
    shrink = BcdiTrad.Shrink(0.1, 1.0, state)
    center = BcdiTrad.Center(state)

    a = Animation()
    # We could run the commands this way, but we want to plot in the middle
    # center * er^500 * (center * er^20 * (shrink * hio)^80)^20 * state
    for i in 1:20
        for j in 1:10
            hio * state
            center * state
            saveAn(state, a)
            shrink * state
        end
        er * state
    end
    for i in 1:1400
        hio * state 
        center * state
        saveAn(state, a)
        shrink * state
    end
    for i in 1:100
        er * state
        center * state
        saveAn(state, a)
    end
    center * state
    saveAn(state, a)

    mov(a, "../results/recon.webm", fps=250)
end

intens, recSupport = simulateDiffraction()
phase(intens, recSupport)
