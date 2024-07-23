using BcdiTrad
using BCDI
using Plots
using FFTW

function phase()
    intensities = round.(Int64, reshape(parse.(Float64, split(readlines("../data/intensities.txt")[1], ",")).^2, 100, 100, 100))

    state = BcdiTrad.State(intensities, trues(size(intensities)))
    er = BcdiTrad.ER()
    hio = BcdiTrad.HIO(0.9)
    shrink = BcdiTrad.Shrink(0.1, 1.0, state)
    center = BcdiTrad.Center(state)

    a = Animation()
    j = 1
    # We could run the commands this way, but we want to plot in the middle
    # center * er^500 * (center * er^20 * (shrink * hio)^80)^20 * state
    for i in 1:1600
        hio * state
        frame(a, heatmap(fftshift(Array(abs.(state.realSpace)))[50,:,:]))
        j += 1
        shrink * state
    end
    for i in 1:100
        er * state
        frame(a, heatmap(fftshift(Array(abs.(state.realSpace)))[50,:,:]))
        j += 1
    end
    center * state

    mov(a, "../results/recon.webm", fps=250)
end

phase()
