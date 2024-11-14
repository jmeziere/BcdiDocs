using BcdiStrain
using BcdiSimulate
using Plots
using Statistics
using FFTW

function saveAnU(state, a)
    p1 = heatmap(fftshift(Array(reshape(state.rho,100,100,100))[1,:,:]))
    p2 = heatmap(fftshift(Array(reshape(state.ux,100,100,100))[1,:,:]))
    p3 = heatmap(fftshift(Array(reshape(state.uy,100,100,100))[1,:,:]))
    p4 = heatmap(fftshift(Array(reshape(state.uz,100,100,100))[1,:,:]))
    p5 = heatmap(fftshift(Array(abs.(state.traditionals[1].realSpace))[1,:,:]))
    p6 = heatmap(fftshift(Array(angle.(state.traditionals[1].realSpace))[1,:,:]))
    frame(a, plot(p1,p2,p5,p3,p4,p6,layout=6,size=(900,400)))
end

function saveAnNU(state, a)
    realSpace = zeros(ComplexF64, 100,100,100)
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
    Gvecs = [
[0.2182028202820282, 0.2182028202820282, -0.21819081908190824], [0.2182028202820282, -0.21819081908190824, 0.2182028202820282], [-0.21819081908190824, 0.2182028202820282, 0.2188028802880288]
    ]
    hRanges = [
        range(Gvecs[1][1]-0.03,Gvecs[1][1]-0.03+100*0.06/101,100),
        range(Gvecs[2][1]-0.03,Gvecs[2][1]-0.03+100*0.06/101,100),
        range(Gvecs[3][1]-0.03,Gvecs[3][1]-0.03+100*0.06/101,100)
    ]
    kRanges = [
        range(Gvecs[1][2]-0.03,Gvecs[1][2]-0.03+100*0.06/101,100),
        range(Gvecs[2][2]-0.03,Gvecs[2][2]-0.03+100*0.06/101,100),
        range(Gvecs[3][2]-0.03,Gvecs[3][2]-0.03+100*0.06/101,100)
    ]
    lRanges = [
        range(Gvecs[1][3]-0.03,Gvecs[1][3]-0.03+100*0.06/101,100),
        range(Gvecs[2][3]-0.03,Gvecs[2][3]-0.03+100*0.06/101,100),
        range(Gvecs[3][3]-0.03,Gvecs[3][3]-0.03+100*0.06/101,100)
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
    x = sign.(x) .* abs.(x).^1.02
    y = Float64.(y)
    y = sign.(y) .* abs.(y).^1.02
    z = Float64.(z)
    z = sign.(z) .* abs.(z).^1.02
    intens, recSupport, GCens, GMaxs, boxSize = BcdiSimulate.atomSimulateDiffraction(x, y, z, hRanges, kRanges, lRanges, rotations, numPhotons)

    recPrimLatt = zeros(3,3)
    for i in 1:3
        GMaxs[i] .*= boxSize
        recPrimLatt[i,:] .= GMaxs[i]
    end
println([GMaxs[i] ./ boxSize for i in 1:3])
for i in 1:3
    maxInd = argmax(intens[i])
    println(maxInd)
    heatmap(log.(intens[i][maxInd[1],:,:]))
    savefig("tmp$i.png")
end
    return intens, recSupport, GMaxs, recPrimLatt, rotations
end

function phase(intens, recSupport, gVecs, recPrimLatt, rotations)
"""
    state = BcdiStrain.State(intens, gVecs, recSupport)
    er = BcdiStrain.ER()
    hio = BcdiStrain.HIO(0.9, state)
    shrink = BcdiStrain.Shrink(0.1, 1.0, state)
    center = BcdiStrain.Center(state)
    mount = BcdiStrain.Mount(0.8, state, recPrimLatt)

    a = Animation()
    # We could run the commands this way, but we want to plot in the middle
    # (mount * center * er^20)^200 * 
    # (mount * center * (shrink * hio)^80)^20 * state
try
    for i in 1:5
        for j in 1:320
            println(i," ",j)
            hio * state
            saveAnU(state, a)
            shrink * state
            center * state
            println(maximum(abs.(state.traditionals[1].realSpace)))
            println(reduce(+,state.traditionals[1].support))
        end
        center * state
        mount * state
    end
    for i in 1:200
        println(i)
        er * state
        center * state
        saveAnU(state, a)
        mount * state
    end
catch e
    println("recon failed")
end
    saveAnU(state, a)
    mov(a, "../results/regstrain.webm", fps=200)
"""
    state = BcdiStrain.State(intens, gVecs, recSupport, highStrain=true)
    eropt = BcdiStrain.EROpt(state)
    er = BcdiStrain.ER()
    hioopt = BcdiStrain.HIOOpt(0.7, state)
    shrink = BcdiStrain.Shrink(0.1, 1.0, state)
    center = BcdiStrain.Center(state)
    mount = BcdiStrain.Mount(0.5, state, recPrimLatt)

    a = Animation()
    # We could run the commands this way, but we want to plot in the middle
    # (mount * center * er^20)^200 *
    # (mount * center * (shrink * hio)^80)^20 * state

    for i in 1:1600
        println(i)
        hioopt * state
        saveAnNU(state, a)
        shrink * state
        println(maximum(abs.(state.traditionals[1].realSpace)))
        println(maximum(abs.(state.traditionals[1].realSpace)))
        println(reduce(+,state.traditionals[1].support))
        eropt * state
        center * state
        mount * state
    end
    for i in 1:100
        println(i)
        eropt * state
        center * state
        mount * state
        saveAnNU(state, a)
    end
    for i in 1:100
        println(i)
        er * state
        center * state
        mount * state
        saveAnNU(state, a)
    end
    mov(a, "../results/highstrain.webm", fps=200)
end

intens, recSupport, GMaxs, recPrimLatt, rotations = simulateDiffraction()
phase(intens, recSupport, GMaxs, recPrimLatt, rotations)
