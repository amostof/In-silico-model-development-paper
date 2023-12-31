βInt = [3.80, 4.18] .* 1E-5
δInt = [3.07, 3.34] .* 1E-5
VgInt = [8.43, 9.46] .* 1E-19
KoInt = [1.86, 2.60] .* 1E-5
KgInt = [2.48, 3.08] .* 1E-8
KlInt = [8.20, 9.23] .* 1E3
cbargInt = [1.65, 2.32] .* 1E0

p = [β, δ, Vg, Ko, Kg, Kl, cbarg]

resultLibrary = extrafineDict
selectedModel = "mmkPmmkPmmkN"
inferredParams = resultLibrary["simulations"][selectedModel]["corrParams"][1]
modelTypes = resultLibrary["simulations"][selectedModel]["modelTypes"]

bFGSParams = extrafineBFGSDict["simulations"][selectedModel]["paramsAllhcat"]
selectedParams = mean(hcat(bFGSParams[2:end]...), dims = 2)
selectedParams[6] = bFGSParams[1][6]

selectedParams[6] = inferredParams[6]
selectedParams[4] = 3.54E-8
selectedParams[5] = 1.77E-9

probCase = ODEProblem(neotissue3EQsCases!, u0s[8, :], (tspan[begin], tspan[end]), selectedParams)

tempPred = solve(probCase, Tsit5())

Plots.plot(tempPred[1,:])

using Distributions
p_dist = [Uniform(βInt[begin], βInt[end]),
            Uniform(δInt[begin], δInt[end]),
            Uniform(VgInt[begin], VgInt[end]),
            Uniform(KoInt[begin], KoInt[end]),
            Uniform(KgInt[begin], KgInt[end]),
            Uniform(KlInt[begin], KlInt[end]),
            Uniform(cbargInt[begin], cbargInt[end])]

prob_funcExpectations(prob, i, repeat) = remake(prob, p = rand.(p_dist))

ensemble_prob = EnsembleProblem(probCase, prob_func = prob_funcExpectations)

ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 100_000)

appendix_save_idxs = [1]
ensemblesol2 = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 100000,
    save_idxs = appendix_save_idxs)


summ = EnsembleSummary(ensemble_sol)
summ2 = EnsembleSummary(ensemblesol2)

# The main line is the median value and the ribbons show the quantile values
Plots.plot(summ)
Plots.plot(summ2)

# idxs does not work
Plots.plot(summ; idxs =  4)

# main line equivalent to Plots.plot(summ2)
Plots.plot(summ2.t./3600, [summ2.med[i][1] for i in eachindex(summ)], legend = false)

plotTitles = ["normal oxygen, high glucose" "normal oxygen, low glucose" "low oxygen, high glucose" "low oxygen, low glucose" ]
plotTitlesCells = ["25 000 initial cells" "50 000 initial cells" "100 000 initial cells" "200 000 initial cells"]

function ensembleICUncertaintyParameters(
    prob :: SciMLBase.AbstractSciMLProblem, 
    p_dist :: Vector{ <: ContinuousUnivariateDistribution}, 
    u0s :: Matrix{ <: Number}; 
    solver = Tsit5(), 
    trajectories = 100_000)

    probFunc(prob, i, repeat) = remake(prob, p = rand.(p_dist))
    # outputDict = Dict{Tuple,SciMLBase.AbstractEnsembleSolution}()
    outputArray = Array{SciMLBase.AbstractEnsembleSolution}(undef, size(u0s,1))
    for i in axes(u0s,1)
        u0 = u0s[i, :]
        newU0Prob = remake(prob, u0 = u0)
        ensembleProb = EnsembleProblem(newU0Prob, prob_func = probFunc)
        
        ensembleSol = solve(ensembleProb, solver, EnsembleThreads(), 
            trajectories = trajectories)

        outputArray[i] = ensembleSol
    end

    return outputArray
end

function plotWithUncertainty(
    ensembleArray :: Array{ <: SciMLBase.AbstractEnsembleSolution},
    dataDict :: Dict{String, Array},
    plotTitles :: Matrix{String}, 
    plotTitlesCells :: Matrix{String}, 
    plotRange :: AbstractRange, 
    states :: Dict{}; 
    colors = Makie.wong_colors(),
    plotrows = 4, 
    plotcolumns = 4, 
    iState = 1)

    yLabel = states[iState]["yLabel"]
    k = states[iState]["k"]
    fig = CairoMakie.Figure(resolution = (1700, 1200))
    subfigs = [CairoMakie.Axis(fig[i, j]) for j in 1:plotcolumns, i in 1:plotrows]

    if plotcolumns != length(plotTitlesCells) || plotcolumns != length(plotTitles)
        return "Error"
    end

    for (index, ensembleSol) in enumerate(ensembleArray[plotRange])
        tempSumm = EnsembleSummary(ensembleSol)
        colIndex = index % plotcolumns == 0 ? 4 : index % plotcolumns
        rowIndex = (index - 1) ÷ plotcolumns + 1
        subfigs[index].title = "$(plotTitlesCells[colIndex]), $(plotTitles[rowIndex])"
        subfigs[index].xlabel = rowIndex == plotrows ? L"Time $(\text{h})$" : ""
        subfigs[index].ylabel = colIndex == 1 ? yLabel : ""
        xdata = tempSumm.t./3600
        CairoMakie.lines!(subfigs[index], xdata, tempSumm.med[iState,:]./k, color = colors[colIndex])
        CairoMakie.band!(subfigs[index], xdata,  tempSumm.qlow[iState,:]./k, tempSumm.qhigh[iState,:]./k, 
            color = (colors[colIndex], 0.1))

        htstepsinferred = dataDict["tsteps"]./3600
        CairoMakie.scatter!(subfigs[index], htstepsinferred, 
            coalesce.(dataDict["observations"][iState,:,plotRange[index]], NaN)./k,  
            color = colors[colIndex], msc = colors[colIndex])
            

        CairoMakie.errorbars!(subfigs[index], htstepsinferred, 
            coalesce.(dataDict["observations"][iState,:,plotRange[index]], NaN)./k,
            coalesce.(dataDict["stds"][iState,:,plotRange[index]], 0)./k, 
            color = colors[colIndex], msc = colors[colIndex], whiskerwidth = 10)
    end
    Makie.linkaxes!(subfigs...)
    fig
end

ensembleArray = ensembleICUncertaintyParameters(probCase, p_dist, dataDict["u0"][begin:end,:];
    trajectories = 50_000)

tempSumm = EnsembleSummary(ensembleArray[5])

plotRange = 1:16
states = Dict{Int32, Dict}(
    1 => Dict{String, Any}("k" => 1E13,
    "yLabel" => L"Cell density $(\times 10^{13} \text{m}^{-2})$"
    ),
    2 => Dict{String, Any}("k" => 1,
    "yLabel" => L"Glucose concentration $(\text{mmol}/\text{l})$"
    ),
    3 => Dict{String, Any}("k" => 1,
    "yLabel" => L"Lactate concentration $(\text{mmol}/\text{l})$"
    )
)
ensemblePlotCell = plotWithUncertainty(ensembleArray, dataDict, plotTitles, plotTitlesCells, plotRange, states;
    colors = [:blue :red :green :purple])
ensemblePlotGlu = plotWithUncertainty(ensembleArray, dataDict, plotTitles, plotTitlesCells, plotRange, states;
    colors = [:blue :red :green :purple], iState = 2)
ensemblePlotLac = plotWithUncertainty(ensembleArray, dataDict, plotTitles, plotTitlesCells, plotRange, states;
    colors = [:blue :red :green :purple], iState = 3)

save(datadir("sims", "figures", "ensemblePlotCell.pdf"), ensemblePlotCell)
save(datadir("sims", "figures", "ensemblePlotGlu.pdf"), ensemblePlotGlu)
save(datadir("sims", "figures", "ensemblePlotLac.pdf"), ensemblePlotLac)

