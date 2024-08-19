using BcdiStrain
using Plots
using FFTW

function saveAn(state, a)
    p1 = heatmap(fftshift(Array(state.rho)[1,:,:]))
    p2 = heatmap(fftshift(Array(state.ux)[1,:,:]))
    p3 = heatmap(fftshift(Array(state.uy)[1,:,:]))
    p4 = heatmap(fftshift(Array(state.uy)[1,:,:]))
    p5 = heatmap(fftshift(Array(abs.(state.traditionals[1].realSpace))[1,:,:]))
    p6 = heatmap(fftshift(Array(angle.(state.traditionals[1].realSpace))[1,:,:]))
    frame(a, plot(p1,p2,p5,p3,p4,p6,layout=6,size=(900,400)))
end

function phase()
    intens = Array{Float64, 3}[]
    gVecs = [[-1.,1,1],[1.,-1,1],[1.,1,-1]]
    primLatt = [-1. 1 1 ; 1 -1 1; 1 1 -1]
    for i in 1:3
        push!(intens, round.(Int64, reshape(parse.(Float64, split(readlines("../data/intensities$(i).txt")[1], ",")), 100, 100, 100)))
    end
    recSupport = [trues(size(intens[1])) for i in 1:length(intens)]

    state = BcdiStrain.State(intens, gVecs, recSupport)
    er = BcdiStrain.ER()
    hio = BcdiStrain.HIO(0.9)
    shrink = BcdiStrain.Shrink(0.1, 1.0, state)
    center = BcdiStrain.Center(state)
    mount = BcdiStrain.Mount(0.5, state, primLatt)

    a = Animation()
    # We could run the commands this way, but we want to plot in the middle
    # (mount * center * er^20)^200 * 
    # (mount * center * (shrink * hio)^80)^20 * state

    for i in 1:20
        for j in 1:80
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
        for j in 1:20
            er * state
            saveAn(state, a)
        end
        center * state
        saveAn(state, a)
        mount * state
        saveAn(state, a)
    end
    saveAn(state, a)
    mov(a, "../results/recon.webm", fps=250)
end

phase()
