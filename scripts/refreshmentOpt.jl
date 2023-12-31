resultLibrary = extrafineDict
selectedModel = "mmkPmmkPmmkN"
inferredParams = resultLibrary["simulations"][selectedModel]["corrParams"][1]
modelTypes = resultLibrary["simulations"][selectedModel]["modelTypes"]

bFGSParams = extrafineBFGSDict["simulations"][selectedModel]["paramsAllhcat"]
selectedParams = mean(hcat(bFGSParams[2:end]...), dims = 2)
selectedParams[6] = inferredParams[6]

refreshSelectiontspan = (tspan[begin], 3*tspan[end])
refreshSelectiontstep = refreshSelectiontspan[begin]:ddt:refreshSelectiontspan[end]
probCase = ODEProblem(neotissue3EQsCases!, u0s[8, :], refreshSelectiontspan, selectedParams)

tempPred = solve(probCase, Tsit5(); saveat=refreshSelectiontstep)

# Plot with each of the observables on a separate scale
cellPlot = Plots.plot(tempPred.t./3600, [tempPred.u[i][1] for i in eachindex(tempPred.u)])
concPlot = Plots.plot(tempPred.t./3600, [tempPred.u[i][2] for i in eachindex(tempPred.u)])
Plots.plot!(tempPred.t./3600,[tempPred.u[i][3] for i in eachindex(tempPred.u)])
display(cellPlot)
display(concPlot)


function condition(u, t, integrator, refreshmentts)
    t ∈ refreshmentts
end


function affect!(integrator)
    integrator.u[2] = integrator.sol.prob.u0[2]
    integrator.u[3] = integrator.sol.prob.u0[3]
end

function forwardSolverRefreshments(prob, p, u0, condition, affect!, refreshmentΔt, tspan, ddt)

    tsteps = tspan[begin]:ddt:tspan[end]
    refreshmentts = refreshmentΔt:refreshmentΔt:tsteps[end]
    bounce_cb = DiscreteCallback((u,t,integrator) -> condition(u,t,integrator, refreshmentts), affect!)

    probBounce = ODEProblem(prob, u0, tspan, p, callback=bounce_cb)
    solve(probBounce, Tsit5(); saveat=tsteps, tstops=refreshmentts)

end


refreshSelectiontspan = (tspan[begin], 5*tspan[end])
refreshmentΔt = 96 * 3600
predBounce = forwardSolverRefreshments(neotissue3EQsCases!, selectedParams, u0s[8, :], 
    condition, affect!, refreshmentΔt, refreshSelectiontspan, ddt)

cellPlotBounce = Plots.plot(predBounce.t./3600/24, 
    [predBounce.u[i][1] for i in eachindex(predBounce.u)],
    labels="Cell density")
concPlotBounce = Plots.plot(predBounce.t./3600/24, 
    [predBounce.u[i][2] for i in eachindex(predBounce.u)],
    labels="Glucose concentration")
Plots.plot!(predBounce.t./3600/24,
    [predBounce.u[i][3] for i in eachindex(predBounce.u)],
    labels="Lactate concentration")

savefig(cellPlotBounce, datadir("sims", "figures", "refreshmentPeriodPop$(refreshmentΔt÷3600).pdf"))
savefig(concPlotBounce, datadir("sims", "figures", "refreshmentPeriodCon$(refreshmentΔt÷3600).pdf"))

display(cellPlotBounce)
display(concPlotBounce)

plotrows, plotcolumns = 3, 4
everyOtherDay = 2
ndays = 24
observableID = 1

function plotDifferentRefreshments(prob, p, u0, condition, affect!, tspan, ddt,
    ndays, everyOtherDay, plotrows, plotcolumns)
    
    fig = CairoMakie.Figure(resolution = (1500, 800))
    subfigs = [CairoMakie.Axis(fig[i, j]) for j in 1:plotcolumns, i in 1:plotrows]
    colors = Makie.wong_colors()
    index = 1
    refreshmentΔts = convert(Array{Int64,1}, collect(1:ndays/everyOtherDay) .* everyOtherDay*(24 * 3600))
    
    for refreshmentΔt in refreshmentΔts
        predBounce = forwardSolverRefreshments(prob, p, u0, 
            condition, affect!, refreshmentΔt, tspan, ddt)
    
        subfigs[index].title = "$(refreshmentΔt÷(24 * 3600)) days"
        subfigs[index].xlabel = L"Time $(\text{day})$"
        subfigs[index].ylabel = L"Cell density $(\times 10^{13} m^{-2})$"
        ydata = [predBounce.u[i][observableID] for i in eachindex(predBounce.u)] / 10^13 
        xdata = predBounce.t./3600/24
        CairoMakie.lines!(subfigs[index], xdata, ydata, color = colors[1])
        refreshTimes = collect(refreshmentΔt:refreshmentΔt:tspan[end])/3600/24
        CairoMakie.vlines!(subfigs[index], refreshTimes, color = colors[2])
        index += 1
    end
    Makie.linkaxes!(subfigs...)
    fig
end

allPeriodsPlot = plotDifferentRefreshments(neotissue3EQsCases!, selectedParams, 
    u0s[8, :], condition, affect!, refreshSelectiontspan, ddt, 
    ndays, everyOtherDay, plotrows, plotcolumns)

save(datadir("sims", "figures", "allPeriodsPlot.pdf"), allPeriodsPlot)
