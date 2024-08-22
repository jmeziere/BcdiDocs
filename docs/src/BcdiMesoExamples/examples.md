# Example 1

## Environment

```julia
[deps]
BcdiCore = "72eb6a3e-ca63-4742-b260-85b05ca6d9e4"
BcdiStrain = "3abd092d-e7bc-4ec6-94c6-c6851986118d"
BcdiMeso = "1ffc817a-885e-4a73-a887-574cb954c7d7"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
```

## Code

```julia
using BcdiCore
using BcdiStrain
using BcdiMeso
using Plots
using FFTW
using LinearAlgebra

function saveAn(rho, ux, uy, uz, inSupp, plotArr, a)
    plotArr[inSupp] .= Array(rho)
    p1 = heatmap(plotArr[50,:,:])
    plotArr[inSupp] .= Array(ux)
    p2 = heatmap(plotArr[50,:,:])
    plotArr[inSupp] .= Array(uy)
    p3 = heatmap(plotArr[50,:,:])
    plotArr[inSupp] .= Array(uz)
    p4 = heatmap(plotArr[50,:,:])
    frame(a, plot(p1,p2,p3,p4,layout=4))
end

function phase()
    intens = Array{Float64, 3}[]
    gVecs = [[-1.,1,1],[1.,-1,1],[1.,1,-1]]
    primLatt = [-1. 1 1 ; 1 -1 1; 1 1 -1]
    for i in 1:3
        push!(intens, round.(Int64, reshape(parse.(Float64, split(readlines("../data/intensities$(i).txt")[1], ",")), 100, 100, 100)))
    end
    recSupport = [trues(size(intens[1])) for i in 1:length(intens)]

    strainState = BcdiStrain.State(intens, gVecs, recSupport)
    er = BcdiStrain.ER()
    hio = BcdiStrain.HIO(0.9)
    shrink = BcdiStrain.Shrink(0.1, 1.0, strainState)
    center = BcdiStrain.Center(strainState)
    mount = BcdiStrain.Mount(0.5, strainState, primLatt)

    a = Animation()
    # We could run the commands this way, but we want to plot in the middle
    (mount * center * er^20)^200 *
    (mount * center * (shrink * hio)^80)^80 * strainState

    A = zeros(3,3)
    for i in 1:3
        _, _, peakLoc = BcdiCore.centerPeak(intens[i], recSupport[i], "corner")
        peakLoc = collect(peakLoc) .+ [1,1,1]
        peakLoc = Int64.(peakLoc)
        h = reshape(parse.(Float64, split(readlines("../data/h$(i).txt")[1], ",")), 100, 100, 100)
        k = reshape(parse.(Float64, split(readlines("../data/k$(i).txt")[1], ",")), 100, 100, 100)
        l = reshape(parse.(Float64, split(readlines("../data/l$(i).txt")[1], ",")), 100, 100, 100)
        peak = [h[peakLoc...],k[peakLoc...],l[peakLoc...]]
        peak ./= [h[1,1,2]-h[1,1,1],k[1,2,1]-k[1,1,1],l[2,1,1]-l[1,1,1]]
        gVecs[i] .= peak
        A[i,:] .= peak
    end

    inSupp = Array(findall(fftshift(strainState.traditionals[1].support)))
    B = zeros(3, reduce(+, strainState.traditionals[1].support))
    B[1,:] .= Array(-fftshift(strainState.ux)[inSupp] .+ fftshift(strainState.uy)[inSupp] .+ fftshift(strainState.uz)[inSupp])
    B[2,:] .= Array(fftshift(strainState.ux)[inSupp] .- fftshift(strainState.uy)[inSupp] .+ fftshift(strainState.uz)[inSupp])
    B[3,:] .= Array(fftshift(strainState.ux)[inSupp] .+ fftshift(strainState.uy)[inSupp] .- fftshift(strainState.uz)[inSupp])

    s = size(intens[1])
    x = zeros(length(inSupp))
    y = zeros(length(inSupp))
    z = zeros(length(inSupp))
    for i in 1:length(inSupp)
        x[i] = 2*pi*(inSupp[i][1]-1)/s[1]
        y[i] = 2*pi*(inSupp[i][2]-1)/s[2]
        z[i] = 2*pi*(inSupp[i][3]-1)/s[3]
    end

    newStrain = A \ B

    support = strainState.traditionals[1].support
    plotArr = zeros(size(support))
    rho = Array(fftshift(strainState.rho)[inSupp])
    ux = -newStrain[1,:]
    uy = -newStrain[2,:]
    uz = -newStrain[3,:]

    mesoState = BcdiMeso.State(
        intens, gVecs, recSupport, x, y, z,
        Array(fftshift(strainState.rho)[inSupp]),
        newStrain[1,:], newStrain[2,:], newStrain[3,:]
    )
    optimizeState1 = BcdiMeso.OptimizeState(mesoState, primLatt, 1)
    optimizeState2 = BcdiMeso.OptimizeState(mesoState, primLatt, 2)
    optimizeState3 = BcdiMeso.OptimizeState(mesoState, primLatt, 3)

    for i in 1:100
        saveAn(mesoState.rho, mesoState.ux, mesoState.uy, mesoState.uz, inSupp, plotArr, a)
        optimizeState1 * mesoState
    end
    for j in 1:100
        saveAn(mesoState.rho, mesoState.ux, mesoState.uy, mesoState.uz, inSupp, plotArr, a)
        optimizeState2 * mesoState
    end
    for k in 1:100
        saveAn(mesoState.rho, mesoState.ux, mesoState.uy, mesoState.uz, inSupp, plotArr, a)
        optimizeState3 * mesoState
    end

    mov(a, "../results/recon.webm", fps=250)
end

phase()
```

## Output

![](Example1/results/recon.webm)
