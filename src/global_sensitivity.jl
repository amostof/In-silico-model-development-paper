######
searchgrid = [1E-1, 1E+1]
bounds = fill(searchgrid,length(p)).*p

f1 = function (p)
  prob1 = remake(prob;p=p)
  solve(prob1,Tsit5();saveat=tsteps)
end

efast_sens = gsa(f1, Sobol(order = [0, 1, 2,]), bounds, samples=50)
# efast_sens = gsa(f1, Sobol(nboot = 10, order = [2,]), bounds, samples=5000)
# efast_sens = gsa(f1, Sobol(order = [2,], nboot = 20), bounds, samples=50000)

# plabels = ["β" "δ" "Vg" "Kl" "cbarg"]
plabels = ["β" "δ" "Vg" "Vo" "Ko" "Kg" "Kl" "cbarg" "cbaro"]
plabels = ["β" "δ" "Vg" "Vo" "Kg" "Kl" "cbarg"]
statevariables = ["Cell Pop" "Glucose" "Lactate"]
plotrows, plotcolumns = 3, 3


function plot_sensitivities(sens, plabels, statevariables, plotrows, plotcolumns)
  fig = CairoMakie.Figure(resolution = (800, 600))

  axes = [CairoMakie.Axis(fig[i, j]) for j in 1:plotcolumns, i in 1:plotrows]

  colors = Makie.wong_colors()
  xdata = sort(repeat(1:(length(tsteps)-1), length(statevariables)))
  dodge = repeat(1:length(statevariables), (length(tsteps)-1))
  for i in eachindex(plabels)
    axes[i].title = plabels[i]
    ydata = reshape(sens.S1[i][1:end-1,2:end], (length(tsteps)-1) * length(statevariables))
    CairoMakie.barplot!(axes[i], xdata, ydata, dodge = dodge, color = colors[dodge])
  end
  title = Label(fig[0,:], "First order Sobol indices")
  elements = [CairoMakie.PolyElement(polycolor = colors[i]) for i in 1:length(statevariables)]
  legend = Legend(fig[plotrows,plotcolumns+1], elements, statevariables)

  return fig
end

fig = plot_sensitivities(efast_sens, plabels, statevariables, plotrows, plotcolumns)
display(fig)

######

searchgridu0 = [1E-1, 1E+1]
boundsu0 = [Vector{Float64}(undef,2) for _ in 1:length(u0)]

searchgridu0 = [1E-1 1E+1;
                1E-1 1E+1;
                1E-1 1E+1;
                1E-1 1E+1;]
      
for i in eachindex(boundsu0)
  boundsu0[i][1] = searchgridu0[i, 1] * u0[i]
  boundsu0[i][2] = searchgridu0[i, 2] * u0[i]
end

### Add parameters and model type

resultLibrary = extrafineDict
selectedModel = "mmkPmmkPmmkN"
inferredParams = resultLibrary["simulations"][selectedModel]["corrParams"][1]
modelTypes = resultLibrary["simulations"][selectedModel]["modelTypes"]
probCase = ODEProblem(neotissue3EQsCases!, u0, tspan, inferredParams)

bFGSParams = extrafineBFGSDict["simulations"][selectedModel]["paramsAllhcat"]
selectedParams = mean(hcat(bFGSParams[2:end]...), dims = 2)
selectedParams[6] = inferredParams[6]
probCase = ODEProblem(neotissue3EQsCases!, u0, tspan, selectedParams)

f2 = function (u0)
  prob1 = remake(probCase;u0=u0)
  solve(prob1,Tsit5();saveat=tsteps)
end
nboot = 40
samples = 40000
sensu0 = gsa(f2, Sobol(order = [0, 1, 2,], nboot = nboot), boundsu0, samples=samples);

output_dictSensu0 = Dict{String, Any}(
"nboot" => nboot,
"samples" => samples, 
"boundsu0" => boundsu0,
"f2" => f2, 
"probCase" => probCase,
"sensu0" => sensu0)

@tagsave(datadir("sims", "Exp", "Sensu0.jld2"), output_dictSensu0)

f2rates = function (u0)
  prob1 = remake(probCase;u0=u0)
  sol = solve(prob1,Tsit5())
  sol(tsteps, Val{1})
end

sensu0rates = gsa(f2rates, Sobol(order = [0, 1, 2,], 
  nboot = 1), boundsu0, samples=samples);

# f2RatesPerCell = function (u0)
#     prob1 = remake(probCase;u0=u0)
#     sol = solve(prob1,Tsit5())
#     sol(tsteps, Val{1}) ./ sol(tsteps)[1,:]
# end

# sensu0RatesPerCell = gsa(f2RatesPerCell, Sobol(order = [0, 1, 2,], 
#   nboot = 1), boundsu0, samples=samples);

reshapeSensConfInt(sens, confInt) = [reshape(confInt[:,i], size(sens[1])) for i in eachindex(sens)]

sensu0_S1_Conf_Int = reshapeSensConfInt(sensu0.S1, sensu0.S1_Conf_Int)
sensu0_ST_Conf_Int = reshapeSensConfInt(sensu0.ST, sensu0.ST_Conf_Int)


ulabels = ["Initial Cell" "Initial Glucose" "Initial Lactate" "Initial Oxygen"]
statevariables = ["Cell Pop" "Glucose" "Lactate"]
plotrows, plotcolumns = 2, 2

plot_sensu0_allTimes = plot_sensitivities(sensu0, ulabels, statevariables, plotrows, plotcolumns)
display(plot_sensu0_allTimes)
save(datadir("sims", "figures", "sensu0_allTimes.pdf"), plot_sensu0_allTimes) 

targetDim = 1
secondAxisDim = 1
excludedDim = 3

function terminal_senstivities(sens, statevariables, ulabels, targetDim, excludedDim)
  fig = CairoMakie.Figure(resolution = (800, 300))
  axes = [CairoMakie.Axis(fig[1, i]) for i in 1:2]

  subtractor = sum(Set([excludedDim]) .<= size(sens.ST[1],1))
  colors = Makie.wong_colors()[1:end .∉ [[excludedDim]]]
  ulabels = ulabels[1:end .∉ [[excludedDim]]]

  xdata = repeat(1:length(statevariables), length(sens.S1)-subtractor)
  dodge = sort(repeat(1:length(sens.S1)-subtractor, length(statevariables)))
  xticks = (1:length(statevariables), vec(statevariables))

  # xdata = sort(repeat(convert(Array{Int64, 1}, tsteps/3600), size(data, 2) - subtractor))
  # dodge = repeat(1:size(data, 2) - subtractor, (size(data, 1)))

  axes[1].title = "First order Sobol indices"
  ydata = hcat([sens.S1[i][1:end-1, end] for i in eachindex(sens.S1)]...)
  ydata = 100 .* ydata[1:end, 1:end .∉ [[excludedDim]]]
  ydata = reshape(ydata, length(dodge))
  CairoMakie.barplot!(axes[1], xdata, ydata, dodge = dodge,
    color = colors[dodge])
  axes[1].xticks = xticks
  axes[1].ytickformat = percent_suffix

  axes[2].title = "Total order Sobol indices"
  ydata = hcat([sens.ST[i][1:end-1, end] for i in eachindex(sens.S1)]...)
  ydata = 100 .* ydata[1:end, 1:end .∉ [[excludedDim]]]
  ydata = reshape(ydata, length(dodge))
  CairoMakie.barplot!(axes[2], xdata, ydata, dodge = dodge, 
    color = colors[dodge])
  axes[2].xticks = xticks
  axes[2].ytickformat = percent_suffix

  elements = [CairoMakie.PolyElement(polycolor = colors[i]) for i in 1:length(ulabels)]
  Legend(fig[1,3], elements, ulabels)

  fig
end
fig2 = terminal_senstivities(sensu0, statevariables, ulabels, targetDim, excludedDim)
display(fig)

save(datadir("sims", "figures", "sensu0_terminal.pdf"), fig2) 

function percent_suffix(values)
  map(values) do v  
      "$(convert(Int32, v))%" 
    end
end

function plot_sens_overtime(sens, statevariables, ulabels, targetDim, excludedDim)

  data = hcat([sens.ST[i][targetDim,:] for i in eachindex(sens.ST)]...)
  statevariable = statevariables[targetDim]

  fig = CairoMakie.Figure(resolution = (800, 300))
  axe = CairoMakie.Axis(fig[1, 1])

  colors = Makie.wong_colors()[1:end .!= excludedDim]
  subtractor = excludedDim <= size(sens.ST[1],1) ? 1 : 0

  xdata = sort(repeat(convert(Array{Int64, 1}, tsteps/3600), size(data, 2) 
    - subtractor))
  dodge = repeat(1:size(data, 2) - subtractor, (size(data, 1)))
  
  axe.title = "Total order Sobol indices on cell population"
  axe.xticks = convert(Array{Int64, 1}, tsteps/3600)
  axe.yticks = [0, round.(10.0 .^ collect(0:2), sigdigits=3)...]
  axe.xlabel = L"\text{Time } (h)"
  ydata = 100 .* vec(transpose(data[:, 1:end .!= excludedDim]))
  axe.yscale = Makie.pseudolog10

  axe.ytickformat = percent_suffix
  CairoMakie.barplot!(axe, xdata, ydata,  dodge = dodge, color = colors[dodge])
  
  elements = [CairoMakie.PolyElement(polycolor = colors[i]) for 
    i in 1:length(ulabels) - subtractor]

  legend = Legend(fig[1,2], elements, ulabels[1:end .!= excludedDim])

  return fig
end

densSensTimePlot = plot_sens_overtime(sensu0, statevariables, ulabels, 1, 1)

densSensTimePlot = plot_sens_overtime(sensu0, statevariables, ulabels, 1, 5)
densSensTimePlot = plot_sens_overtime(sensu0rates, statevariables, ulabels, 1, 3)
save(datadir("sims", "figures", "densSensTimePlot.pdf"), densSensTimePlot) 

function plot_sens_overtime_two_axes(sens, ulabels, targetDim, secondAxisDim, excludedDim)
  data = hcat([sens.ST[i][targetDim,:] for i in eachindex(sens.ST)]...)

  fig = CairoMakie.Figure(resolution = (800, 300))
  axL = CairoMakie.Axis(fig[1, 1])
  axR = CairoMakie.Axis(fig[1, 1], yaxisposition = :right)
  hidespines!(axR)
  hidexdecorations!(axR)

  colors = Makie.wong_colors()[1:end .∉ [[excludedDim]]]
  subtractor = sum(Set([excludedDim]) .<= size(sens.ST[1],1))

  xdata = sort(repeat(convert(Array{Int64, 1}, tsteps/3600), size(data, 2) - subtractor))
  dodge = repeat(1:size(data, 2) - subtractor, (size(data, 1)))

  # ydata = vec(transpose(data[:, 1:end .∉ [[excludedDim]]]))
  ydata = data[:, 1:end .∉ [[excludedDim]]]

  axR.ytickcolor = Makie.wong_colors()[secondAxisDim]
  axR.yticklabelcolor = Makie.wong_colors()[secondAxisDim]

  axR.ytickcolor = Makie.wong_colors()[secondAxisDim]
  axL.title = "Total order Sobol indices on cell population"
  axL.xticks = convert(Array{Int64, 1}, tsteps/3600)
  axL.xlabel = L"\text{Time } (h)"
  # axL.yscale = Makie.pseudolog10
  # Makie.ylims!(0.001, 1) 
  ydata1 = vec(transpose(map(x -> x[2] ∉ [secondAxisDim], CartesianIndices(ydata)) .* ydata))
  ydata2 = vec(transpose(map(x -> x[2] ∈ [secondAxisDim], CartesianIndices(ydata)) .* ydata))
  CairoMakie.barplot!(axL, xdata, ydata1, dodge = dodge, color = colors[dodge])
  CairoMakie.barplot!(axR, xdata, ydata2, dodge = dodge, color = colors[dodge])

  elements = [CairoMakie.PolyElement(polycolor = colors[i]) for i in 1:length(ulabels)-subtractor]
  legend = Legend(fig[1,2], elements, ulabels[1:end .∉ [[excludedDim]]])

  fig
end
densSensTimePlot = plot_sens_overtime_two_axes(sensu0rates, ulabels, targetDim, secondAxisDim, excludedDim)
