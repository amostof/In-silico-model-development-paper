using DrWatson
@quickactivate "In-silico-model-development-paper"

using Unitful, DataFrames, CSV, DelimitedFiles
using DifferentialEquations, DiffEqFlux, Random, LinearAlgebra
using DiffEqBase: UJacobianWrapper
using Plots, StatsPlots, LaTeXStrings, StatsBase
using CairoMakie, Statistics, Distributions
using LatinHypercubeSampling, GlobalSensitivity
using SciMLSensitivity, ForwardDiff, Zygote, Optimisers
using Optimization, OptimizationOptimJL, OptimizationPolyalgorithms, OptimizationMultistartOptimization, OptimizationOptimisers, OptimizationNLopt
using ParameterHandling, LikelihoodProfiler, NLopt, Clustering, MultivariateStats, UMAP
using Distances, ProfileLikelihood

include(srcdir("read_data.jl"))
include(srcdir("loss_functions.jl"))
include(srcdir("forward_solvers.jl"))
include(srcdir("inverse_solvers.jl"))
include(srcdir("call_backs.jl"))
include(srcdir("sampling.jl"))
include(srcdir("synthetic_generator.jl"))
include(srcdir("physics.jl"))
include(srcdir("utils.jl"))
include(srcdir("output_messages.jl"))
include(srcdir("multi_guess.jl"))
include(srcdir("data_splitting.jl"))

# Initial conditions
initialCells = 50000.
celldia = 15.0u"μm" 
wellArea = 9.6u"cm^2"
n₀ = initialCells / wellArea / celldia * 1.0u"mol"
cg₀ = 5.2u"mmol/l"
cl₀ = 1.6125u"mmol/l"
co₀ = incubatorOxygen()
u0 = [n₀, cg₀, cl₀, co₀[1]]
u0s = [n₀/2 cg₀ cl₀ co₀[1];
      n₀   cg₀ cl₀ co₀[1];
      2*n₀ cg₀ cl₀ co₀[1];
      4*n₀ cg₀ cl₀ co₀[1];
      n₀/2 4*cg₀ cl₀ co₀[1];
      n₀   4*cg₀ cl₀ co₀[1];
      2*n₀ 4*cg₀ cl₀ co₀[1];
      4*n₀ 4*cg₀ cl₀ co₀[1];
      n₀/2 cg₀ cl₀ co₀[2];
      n₀   cg₀ cl₀ co₀[2];
      2*n₀ cg₀ cl₀ co₀[2];
      4*n₀ cg₀ cl₀ co₀[2];
      n₀/2 4*cg₀ cl₀ co₀[2];
      n₀   4*cg₀ cl₀ co₀[2];
      2*n₀ 4*cg₀ cl₀ co₀[2];
      4*n₀ 4*cg₀ cl₀ co₀[2];]


u0 = dedimension(u0)
u0s = dedimension(u0s)


# Optimized parameters
β = ustrip(uconvert(u"s/s^2", 3.79E-05u"1/s"))
δ = ustrip(uconvert(u"s/s^2", 5.88E-06u"1/s"))
Vg = ustrip(uconvert(u"s/s^2", 2.95E-18u"1/s"))
Vo = ustrip(uconvert(u"s/s^2", 2.00E-19u"1/s"))
# Vo = ustrip(uconvert(u"s/s^2", 0.0u"1/s"))
Ko = ustrip(uconvert(u"mol/m^3", 0.001u"mol/m^3"))
Kg = ustrip(uconvert(u"mol/m^3", 0.001u"mol/m^3"))
Kl = ustrip(uconvert(u"mol/m^3", 0.001u"mol/m^3"))
cbarg = ustrip(uconvert(u"mol/m^3", 103.6u"mol/m^3"))
cbaro = ustrip(uconvert(u"mol/m^3", 6.66E-9u"mol/m^3"))
  
# Simulation interval and intermediary points
totalt = ustrip(uconvert(u"s", 114.0u"hr"))
ddt = ustrip(uconvert(u"s", 12.0u"hr"))
ddtoutput = ustrip(uconvert(u"s", 10.0u"minute"))
initialTime = ustrip(uconvert(u"s", 0.0u"hr"))
initialSample = ustrip(uconvert(u"s", 6.0u"hr"))
tspan = (initialTime, totalt)
tsteps = initialSample:ddt:totalt
save_idxs = [1, 2, 3]

# Neotissue growth parameters
# p = [β, δ, Vg, Kl, cbarg,]
p = [β, δ, Vg, Vo, Ko, Kg, Kl, cbarg, cbaro]
p = [β, δ, Vg, Ko, Kg, Kl, cbarg]

latexplabels = [L"β" L"δ" L"V_g" L"V_o" L"K_o" L"K_g" L"K_l" L"\bar{c}_g" L"\bar{c}_o"]	
latexplabels = [L"β" L"δ" L"V_g" L"K_o" L"K_g" L"K_l" L"\bar{c}_g"]

# p = [β,δ]
p = dedimension(p)


u0sLabels = ["II, 25k" "II, 50k" "II, 100k" "II, 200k";;
              "I, 25k" "I, 50k"   "I, 100k"  "I, 200k";;
          "III, 25k" "III, 50k" "III, 100k" "III, 200k";;
            "IV, 25k" "IV, 50k"   "IV, 100k" "IV, 200k"]

# Setup the ODE problem, then solve
prob = ODEProblem(neotissue3EQsCases!, u0, tspan, p)

# odesolver = AutoTsit5(Rosenbrock23())
odesolver = Rosenbrock23()
# odesolver = Rodas5()

nObeservationsPerPoint = 9 

modelCases = [:mmkP, :mmkP, :mmkN]

oCases = [:const, :linear, :mmkP]
gCases = [:const, :linear, :mmkP]
lCases = [:const, :linear, :mmkN]

allCases = hcat([[o, g, l] for o in oCases, g in gCases, l in lCases]...)

modelTypes = allCases[:, end-2]

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
sim = solve(ensemble_prob, odesolver, EnsembleThreads(), 
            trajectories=size(u0s)[1], saveat=tsteps, save_idxs=save_idxs)

probCases = ODEProblem(neotissue3EQsCases!, u0, tspan, p)

ensemble_prob = EnsembleProblem(probCases, prob_func=prob_func)
simCases = solve(ensemble_prob, odesolver, EnsembleThreads(), 
            trajectories=size(u0s)[1], saveat=tsteps, save_idxs=save_idxs)

# Plot the solution
gr(markerstrokewidth=0)
layout = @layout [a; b c]
htsteps = tsteps / 3600.0
p1 = Plots.plot(htsteps,[sim[i][1,:] for i in 1:length(sim)],lw=3)
p2 = Plots.plot(htsteps, [sim[i][2, :] for i in 1:length(sim)], lw = 3)
p3 = Plots.plot(htsteps,[sim[i][3,:] for i in 1:length(sim)],lw=3)
display(Plots.plot(p1, p2, p3, layout = layout, label = u0sLabels))

printer = false
plotter = false
rng = MersenneTwister(1234)
noiselevel = 0.2

# noise_multiplier = 1.0 .+ append!(noiselevel.* ones(size(sim, 1)-1), 0) .* randn(rng, size(sim))
noise_multiplier = 1.0 .+ noiselevel .* randn(rng, size(sim))
noisydata = sim .* noise_multiplier 

correctness = 1E-12 # 1E-10
initialp = ones(length(p)) * correctness
initialp = initialp .* p

maxiters = 1000

plotloss = []
initialp = dedimension(initialp)

result_ode = PolyOptModified(loss, noisydata, initialp, callback, maxiters, prob, prob_func; opt = Optimisers.Adam(0.1), 
              BFGSsolver = false, save_idxs = save_idxs, useLikelihood = true)

display(Plots.plot(plotloss))
# display(plot(plotloss, yscale=:log10))

finalp = exp.(result_ode.u)

output_synthetic_one_run(finalp)
###


### Unimportant: Just to check how the loss changes given different initial 
# lossAtPoint = []
# for noiselevel in 0.01:0.01:1
#   noisydata = sim .* (1.0 .+ noiselevel .* randn(rng, size(sim))) 
#   append!(lossAtPoint, loss(initialp, noisydata, prob, prob_func; logscale = false, 
#       save_idxs = save_idxs, onlyLoss = true, useLikelihood = false))
# end
# Plots.plot(lossAtPoint)

###

# include("synthetic_inference.jl")
# include(srcdir("profile_likelihood.jl"))
include("read_split_datasets.jl")

loss_over_number(params, input, prob) = lossRealData(params, input, prob, prob_funcData;
  onlyLoss = true, logscale = false)  ./ sum(input["observations"] .!== missing)


oCases = [:const, :linear, :mmkP]
gCases = [:const, :linear, :mmkP]
lCases = [:const, :linear, :mmkN]

allCases = hcat([[o, g, l] for o in oCases, g in gCases, l in lCases]...)

searchgrid = [1E-7, 1E+7]
bounds = fill(searchgrid,length(p)).*p

nguess = 300
initialps = latinCube(bounds, length(p), nguess)
println("done with creating initial guesses")
output_dict = Dict{String, Any}("p" => p,
  "nps" => length(p),
  "nguess" => nguess, 
  "searchgrid" => searchgrid,
  "initialps" => initialps,
  "dataDict" => dataDict,
  "trainingDict" => trainingDict,
  "validationDict" => validationDict,
  "testingDict" => testingDict,
  "simulations" => Dict{String, Any}())


for case in axes(allCases, 2)

  modelTypes = allCases[:, case]
  probCase = ODEProblem(neotissue3EQsCases!, u0, tspan, p)
  paramsAll, plotlossAll, lossesAll = diffInitialGuessData(lossRealData, initialps, trainingDict, probCase, prob_funcData; alg = :PolyOpt, BFGS = false);
  # paramsAll, plotlossAll, lossesAll = diffInitialGuessData(loss, initialps, noisydata; alg = :PolyOpt, BFGS = false);

  paramsAllhcat =  permutedims(hcat(paramsAll...))
  lossesAllhcat =  permutedims(hcat(lossesAll...))

  bestGuesses = [argmin(lossesAllhcat, dims = 1)[i][1] for i in 1:size(lossesAllhcat)[2]]
  bestLosses = [minimum(lossesAllhcat, dims = 1)[i][1] for i in 1:size(lossesAllhcat)[2]]
  corrParams = [paramsAllhcat[bestGuesses[i],i] for i in 1:size(paramsAllhcat)[2]]

  # Loss values on different data sets
  totalLossOverNo = loss_over_number(corrParams[1], dataDict, probCase)
  trainLossOverNo = loss_over_number(corrParams[1], trainingDict, probCase)
  validLossOverNo = loss_over_number(corrParams[1], validationDict, probCase)
  testLossOverNo = loss_over_number(corrParams[1], testingDict, probCase)

  println("\n************************************* \n Analysis complete. ")
  println("Ground truth: $(p)\nNumber of parameter guesses: $(nguess)\nSearch Grid: $(searchgrid)")
  println("Final parameters: $(round.(mean(corrParams), sigdigits=4))")
  println("Loss: $(round(mean(bestLosses), sigdigits=3))")
  println("tsteps: $(ddt/3600)h")
  println("Error: $(round(norm(mean(corrParams) - p) ./ norm(p) .* 100, sigdigits=3))%")
  println("Error per parameter: $(round.((mean(corrParams) - p) ./ (p) .* 100, sigdigits=3))%")
  print("Mean total loss: $(round(totalLossOverNo, sigdigits = 4)), ")
  println("Mean train loss: $(round(trainLossOverNo, sigdigits = 4))")
  print("Mean valid loss: $(round(validLossOverNo, sigdigits = 4)), ")
  println("Mean test loss: $(round(testLossOverNo, sigdigits = 4))")

  temp_dict = Dict{String, Any}(
    "paramsAllhcat" => paramsAllhcat,
    "lossesAllhcat" => lossesAllhcat,
    "bestGuesses" => bestGuesses,
    "bestLosses" => bestLosses,
    "corrParams" => corrParams,
    "totalLossOverNo" => totalLossOverNo,
    "trainLossOverNo" => trainLossOverNo,
    "validLossOverNo" => validLossOverNo,
    "testLossOverNo" => testLossOverNo,
    "modelTypes" => modelTypes)

  output_dict["simulations"][string(string.(modelTypes)...)] = temp_dict
end

@tagsave(datadir("sims", "Exp", savename(output_dict, "jld2")), output_dict)

for (modelname, mydic)  in simulations # output_dict["simulations"]
  println("The validation loss for model $(modelname) is equal to $(mydic["validLossOverNo"])")
  # println("The validation loss for model $mydic is equal to $(mydic["validLossOverNo"])")
end

function dictionary2DataFrame(simulations)
  resultsSummary = Dict()
  resultsSummary["modelType"] = collect(keys(simulations))
  resultsSummary["validLossOverNo"] = [collect(values(simulations))[i]["validLossOverNo"] for i in 1:length(simulations)]
  resultsSummary["totalLossOverNo"] = [collect(values(simulations))[i]["totalLossOverNo"] for i in 1:length(simulations)]
  resultsSummary["trainLossOverNo"] = [collect(values(simulations))[i]["trainLossOverNo"] for i in 1:length(simulations)]
  resultsSummary["testLossOverNo"] = [collect(values(simulations))[i]["testLossOverNo"] for i in 1:length(simulations)]
  resultsSummarydf = DataFrame(resultsSummary)
  numberofParams = 4 .+ length.(findall.("mmk",resultsSummarydf[!, "modelType"]))
  doubleNegativeLogLikelihood = resultsSummarydf[!, "validLossOverNo"] .*  sum(validationDict["observations"] .!== missing) * nObeservationsPerPoint
  numberofAllValidationPoints = sum(validationDict["observations"] .!== missing) * nObeservationsPerPoint
  resultsSummarydf[!, "AIC"] = @. 2*numberofParams + doubleNegativeLogLikelihood
  resultsSummarydf[!, "BIC"] = @. log(numberofAllValidationPoints) * numberofParams + 
    doubleNegativeLogLikelihood
  resultsSummarydf[!, "AICc"] = @. 2*numberofParams + (2*numberofParams) *
    (numberofParams+1)/(numberofAllValidationPoints-numberofParams-1) + doubleNegativeLogLikelihood
  return resultsSummarydf
end

resultsSummarydf = dictionary2DataFrame(simulations)

CSV.write(datadir("sims", "tables", "inferredModels.csv"), resultsSummarydf)

resultsExtraSummarydf = dictionary2DataFrame(extrafineDict["simulations"])
resultsExtraBFGSSummarydf = dictionary2DataFrame(extrafineBFGSDict["simulations"])



# Multiply the loss value by this number to get the likelihood loss value
nObeservationsPerPoint = 9 
###
dummyParams = simulations["constconstmmkN"]["corrParams"][1]
dummyProb = ODEProblem(neotissue3EQsCases!, u0, tspan, p)
display(loss_over_number(dummyParams, dataDict, dummyProb))

function plotModelsValidLossOnLinears(df, column; final = 27, ylabel="Validation loss")
  sortedperm = sortperm(df[!, column])[begin:final]
  groups = length.(findall.("linear",df[!, "modelType"]))[sortedperm]
  markers = intersect(Plots._shape_keys, Plots.supported_markers())
  markershape = [markers[groups[i]+1] for i in eachindex(groups)]
  
  Plots.scatter(sort(df[!, column])[begin:final], 
    group=groups, markershape = markershape,
    legend = :topleft, xlabel="Model rank", ylabel=ylabel)
  
end


histogram(log.(sqrt.(lossesAllhcat)), legend = false, xlabel = "Log loss", bins = 50, size = (400, 300))
savefig(datadir("sims", "figures", "ParamsHisto.pdf"))

# Waterfall plot
display(Plots.plot(log.(sort(lossesAllhcat, dims = 1)), legend = false, xlabel = "Loss index", ylabel = "Log loss", size = (400, 300)))
Plots.savefig(datadir("sims", "figures", "waterfall.pdf"))

plotModelsValidLossOnLinears(resultsSummarydf, "validLossOverNo")
plotAIC = plotModelsValidLossOnLinears(resultsSummarydf, "AIC"; ylabel = "AIC")
plotBIC = plotModelsValidLossOnLinears(resultsSummarydf, "BIC"; ylabel = "BIC")
plotAICc = plotModelsValidLossOnLinears(resultsSummarydf, "AICc"; ylabel = "AICc")

savefig(plotAIC, datadir("sims", "figures", "AIC.pdf")) 
savefig(plotBIC, datadir("sims", "figures", "BICAllModels.pdf")) 
savefig(plotAICc, datadir("sims", "figures", "AICc.pdf")) 

badModels = ["constconstconst", "linearconstconst", "mmkPconstconst", 
              "constlinearconst", "constmmkPconst"]

resultsSummaryStatSelectdf = resultsSummarydf[.~([i in badModels for i in 
                                                resultsSummarydf.modelType]), :]

plotAIC = plotModelsValidLossOnLinears(resultsSummaryStatSelectdf, "AIC"; final = 22, ylabel = "AIC")
plotBIC = plotModelsValidLossOnLinears(resultsSummaryStatSelectdf, "BIC"; final = 22, ylabel = "BIC")
plotAICc = plotModelsValidLossOnLinears(resultsSummaryStatSelectdf, "AICc"; final = 22, ylabel = "AICc")                                                
savefig(plotAIC, datadir("sims", "figures", "AICStatSelect.pdf")) 
savefig(plotBIC, datadir("sims", "figures", "BICAllModelsStatSelect.pdf")) 
savefig(plotAICc, datadir("sims", "figures", "AICcStatSelect.pdf")) 



resultsSummarydfServer = dictionary2DataFrame(alldata2["simulations"])
plotModelsValidLossOnLinears(resultsSummarydfServer, "validLossOverNo"; final = 8)
plotAIC = plotModelsValidLossOnLinears(resultsSummarydfServer, "AIC"; final = 8, ylabel = "AIC")
plotBIC = plotModelsValidLossOnLinears(resultsSummarydfServer, "BIC"; final = 8, ylabel = "BIC")
plotAICc = plotModelsValidLossOnLinears(resultsSummarydfServer, "AICc"; final = 8, ylabel = "AICc")

selectedModel = "mmkPmmkPmmkN"

histogram(log.(alldata["simulations"][selectedModel]["lossesAllhcat"]./2), legend = false, xlabel = L"\log \ell", bins = 50, size = (400, 300))
savefig(datadir("sims", "figures", "$(selectedModel)_ParamsHisto.pdf"))

display(Plots.plot(log.(sort(alldata2["simulations"][selectedModel]["lossesAllhcat"]./2, dims = 1)), legend = false, xlabel = "Loss index", ylabel = L"\log \ell", size = (400, 300)))
Plots.savefig(datadir("sims", "figures", "$(selectedModel)_waterfall.pdf"))

new_losses = deepcopy(alldata2["simulations"][selectedModel]["lossesAllhcat"])
new_losses = vcat(new_losses, alldata["simulations"][selectedModel]["lossesAllhcat"])
display(Plots.plot(log.(sort(new_losses./2, dims = 1)), 
  legend = false, xlabel = "Loss index", ylabel = L"\log \ell", 
  size = (400, 300), linewidth=5))
Plots.savefig(datadir("sims", "figures", "$(selectedModel)_waterfall.pdf"))

histogram(log.(new_losses./2), legend = false, xlabel = L"\log \ell", bins = 50, size = (400, 300))
savefig(datadir("sims", "figures", "$(selectedModel)_ParamsHisto.pdf"))


histogram(log.(extrafineDict["simulations"]["mmkPmmkPconst"]["lossesAllhcat"]./2), legend = false, xlabel = L"\log \ell", bins = 50, size = (400, 300))
display(Plots.plot(log.(sort(extrafineDict["simulations"]["mmkPmmkPconst"]["lossesAllhcat"]./2, dims = 1))[begin:100], legend = false, xlabel = "Loss index", ylabel = L"\log \ell", size = (400, 300)))


paramsAllMat = permutedims(hcat(paramsAllhcat...))

paramAllUnit = standardize(UnitRangeTransform, log.(paramsAllMat), dims = 1)	
Plots.plot([histogram(paramAllUnit[:,i], title=latexplabels[i], bins = 30) for i in 1:size(paramsAllMat,2)]..., size=(1000, 500), legend = false)	
Plots.savefig(datadir("sims", "figures", "histograms.pdf"))


paramsallhcat = alldata2["simulations"]["mmkPmmkPconst"]["paramsAllhcat"]
lossesAllhcat = alldata2["simulations"]["mmkPmmkPconst"]["lossesAllhcat"]./2


modelnames = ["mmkPmmkPmmkN", "mmkPmmkPconst", "mmkPconstmmkN", "constmmkPmmkN", 
  "constconstmmkN"]

function plotUnitParamHistograms(modelname, dataset; cutoff=100)
  paramsallhcat = dataset["simulations"][modelname]["paramsAllhcat"]
  lossesAllhcat = dataset["simulations"][modelname]["lossesAllhcat"]./2
  paramsAllMat = permutedims(hcat(paramsallhcat...))
  paramAllUnit = standardize(UnitRangeTransform, log.(paramsAllMat), dims = 1)	

  lossesAllselect = lossesAllhcat[sortperm(vec(lossesAllhcat))][begin:cutoff]
  paramsAllUnitselect = paramAllUnit[sortperm(vec(lossesAllhcat)),:][begin:cutoff,:]

  b_range = range(0, 1, length=21)
  return Plots.plot([histogram(paramsAllUnitselect[:,i], bins = b_range, title=latexplabels[i]) 
          for i in 1:size(paramAllUnit,2)]..., size=(1000, 500), legend = false)	
end

for modelname in modelnames
  temp_plot = plotUnitParamHistograms(modelname, alldata)
  display(temp_plot)
  temp_plot = plotUnitParamHistograms(modelname, extrafineDict)
  display(temp_plot)
end

plotUnitParamHistograms(selectedModel, extrafineDict)

Plots.savefig(datadir("sims", "figures", "mmkPmmkPmmkN_histograms.pdf"))

paramsStand = standardize(ZScoreTransform, log.(paramsAllMat), dims = 1)
M = fit(PCA, paramsStand'; maxoutdim = 2)
params2D = predict(M, paramsStand')

sils = []
for nclusters in 1:40
  R = kmeans(paramsStand', 20; maxiter=200, display=:iter);
  distances = pairwise(SqEuclidean(), paramsStand');
  append!(sils, mean(silhouettes(assignments(R), counts(R), distances)))
end
display(Plots.plot(1:40, sils))

R = kmeans(paramsStand', 7; maxiter=200, display=:iter);

Plots.scatter(params2D[1,:], params2D[2,:], marker_z = R.assignments, color=:lightrainbow)
Plots.scatter(params2D[1,:], params2D[2,:], marker_z = lossesAllhcat, color=:bwr)

mythreshold =  6.5 #4 # 3.295 # 3.285
lossesAllselect = lossesAllhcat[log.(lossesAllhcat) .< mythreshold];
paramsAllselect = paramsAllhcat[log.(lossesAllhcat) .< mythreshold];

paramsAllselectMat = permutedims(hcat(paramsAllselect...))

histogram(log.(lossesAllselect), legend = false, xlabel = "Loss", bins = 50)

Plots.plot([histogram(paramsAllselectMat[:,i], title=latexplabels[i], bins = 50) for i in 1:size(paramsAllselectMat,2)]..., size=(1000, 500), legend = false)	
confIntervals = [median_interval(paramsAllselectMat[:,i], 0.95) for i in axes(paramsAllselectMat, 2)]
Plots.plot([histogram(log.(paramsAllselectMat[:,i]), title=latexplabels[i], bins = 50) for i in 1:size(paramsAllselectMat,2)]..., size=(1000, 500), legend = false)	

paramAllUnit = standardize(UnitRangeTransform, log.(paramsAllselectMat), dims = 1)	
Plots.plot([histogram(paramAllUnit[:,i], title=latexplabels[i], bins = 50) for i in 1:size(paramsAllselectMat,2)]..., size=(1000, 500), legend = false)	
Plots.savefig(datadir("sims", "figures", "histogramsWThreshold.pdf"))

paramsStand = standardize(ZScoreTransform, paramsAllselectMat, dims = 1)
M = fit(PCA, paramsStand'; maxoutdim = 2)
params2D = predict(M, paramsStand')

ncounts = []
most_populated = []
nclusters = 4
for itry in 1:1000
  sils = []
  populated = []
  for kclusters in 2:nclusters
    R = kmeans(paramsStand', kclusters; maxiter=200, display=:iter);
    distances = pairwise(SqEuclidean(), paramsStand');
    append!(sils, mean(silhouettes(assignments(R), counts(R), distances)))
    append!(populated, max(R.counts...))
  end
  # display(Plots.plot(2:nclusters, sils))
  append!(ncounts, argmax(sils) + 1)
  append!(most_populated, populated)
end

histogram(ncounts, bins = 11)
R = kmeans(paramsStand', mode(ncounts); maxiter=200, display=:iter);

display(Plots.plot(log.(sort(lossesAllselect)), legend = false, xlabel = "Loss index", ylabel = "Log RSS"))

Plots.scatter(log.(lossesAllselect), color = R.assignments, ms=3, ma=0.8, legend = false, xlabel = "Loss index", ylabel = "Log RSS")

nonMain = R.assignments .!= argmax(R.counts)
lossesAllnonMain = lossesAllselect[nonMain]

Plots.scatter(log.(lossesAllnonMain), color = R.assignments[nonMain], ms=3, ma=0.8, legend = false, xlabel = "Loss index", ylabel = "Log RSS")


paramsStandGuessAndInferred = standardize(ZScoreTransform, vcat(initialps', paramsAllMat), dims = 1)
MGuessAndInferred = fit(PCA, paramsStandGuessAndInferred'; maxoutdim = 2)
params2DGuessAndInferred = predict(MGuessAndInferred, paramsStandGuessAndInferred')

params2DInferred = params2DGuessAndInferred[:, begin:nguess]
params2DGuess = params2DGuessAndInferred[:, nguess + 1:end]

Plots.scatter(params2DInferred[1,:], params2DInferred[2,:], marker_z = lossesAllhcat, color=:bwr)
Plots.scatter!(params2DGuess[1,:], params2DGuess[2,:], color=:matter)


params2DGuessAndInferredUmap = umap(paramsStandGuessAndInferred', 2)

params2DInferredUmap = params2DGuessAndInferredUmap[:,begin:nguess]
params2DGuessUmap = params2DGuessAndInferredUmap[:, nguess + 1:end]

Plots.scatter(params2DInferredUmap[1,:], params2DInferredUmap[2,:], marker_z = lossesAllhcat, color=:lightrainbow, markersize = 4)
Plots.scatter!(params2DGuessUmap[1,:], params2DGuessUmap[2,:], color=:matter, )

topn = 35
topparams2DInferredUmap = params2DInferredUmap[:,sortperm(vec(lossesAllhcat))[begin:topn]]
plotUMAP = Plots.scatter(params2DInferredUmap[1,:], params2DInferredUmap[2,:], marker_z = lossesAllhcat, color=:matter, markersize = 5, )
Plots.scatter!(plotUMAP, topparams2DInferredUmap[1,:], topparams2DInferredUmap[2,:], markershape = :diamond, color = :green, markersize = 8, legend = false)
Plots.savefig(plotUMAP, datadir("sims", "figures", "ParamsUMAP.pdf"))



histogram(log.(lossesAllhcat), legend = false, xlabel = "Loss", bins = 50)
savefig(datadir("sims", "figures", "ParamsHisto.pdf"))

topk = 2

inferredsim = forwardSolver(hcat(paramsAllselect[sortperm(lossesAllselect)[topk]]...), dataDict["tsteps"], dataDict["u0"])

inferredsim4 = forwardSolver(exp.(result_ode.u), dataDict["tsteps"]/10, dataDict["u0"])

# inferredloss, inferredsim, _ = lossRealData(log.(corrParams[1]), dataDict);

topValues = 20
topParams = paramsAllselectMat[sortperm(lossesAllselect)[begin:topValues],:]

p_arr1 = Array{Any}(nothing, 7)
for i in 1:size(topParams, 2)
  Plots.plot(topParams[:,i])
  p_arr1[i] = hline!([mean(topParams[:, i]) - std(topParams[:, i]), 
    mean(topParams[:, i]) + std(topParams[:, i])])
end
Plots.plot(p_arr1...)

plotlabelsSilico = ["In silico, 25 000 initial cells" "In silico, 50 000 initial cells" "In silico, 100 000 initial cells" "In silico, 200 000 initial cells"]
plotlabelsVivo = ["In vitro, 25 000 initial cells" "In vitro, 50 000 initial cells" "In vitro, 100 000 initial cells" "In vitro, 200 000 initial cells"]

plotTitles = ["normal oxygen, high glucose" "normal oxygen, low glucose" "low oxygen, high glucose" "low oxygen, low glucose" ]

reportlayout = @layout [a b c]

for iExp in 1:nExp
  mycolors = [:blue :red :green :purple]
  plotRange = (iExp - 1) * numberOfReplicates + 1 : iExp * numberOfReplicates
  p1 = Plots.plot(htstepsinferred,[inferredsim[i][1,:] for i in plotRange],lw=3, color = mycolors, 
    label = plotlabelsSilico, xlabel = L"Time $(h)$", ylabel = L"Cell density $(1/m^2)$", title = "Exp$(iExp), $(plotTitles[iExp])")
  Plots.scatter!(p1, htstepsinferred, dataDict["observations"][1,:,plotRange],  yerr = coalesce.(dataDict["stds"][1,:,plotRange], 0), 
    color = mycolors, msc = mycolors, legend = :topleft, label = plotlabelsVivo)
  p2 = Plots.plot(htstepsinferred,[inferredsim[i][2,:] for i in plotRange],lw=3,  color = mycolors, 
    label = plotlabelsSilico, xlabel = L"Time $(h)$", ylabel = L"Glucose concentration $(mmol/l)$", title = "Exp$(iExp), $(plotTitles[iExp])")
  Plots.scatter!(p2, htstepsinferred, dataDict["observations"][2,:,plotRange],  yerr = coalesce.(dataDict["stds"][2,:,plotRange], 0), 
    color = mycolors, msc = mycolors, legend = :bottomleft, label = plotlabelsVivo)
  p3 = Plots.plot(htstepsinferred,[inferredsim[i][3,:] for i in plotRange],lw=3, color = mycolors, 
    label = plotlabelsSilico, xlabel = L"Time $(h)$", ylabel = L"Lactate concentration $(mmol/l)$", title = "Exp$(iExp), $(plotTitles[iExp])")
  Plots.scatter!(p3, htstepsinferred, dataDict["observations"][3,:,plotRange],  yerr = coalesce.(dataDict["stds"][3,:,plotRange], 0), 
    color = mycolors, msc = mycolors, legend = :topleft, label = plotlabelsVivo)
  display(Plots.plot(p1, p2, p3, layout = (1, 3), size=(2000, 400)))
    # Plots.savefig(p1, "output/inferredCellExp$(iExp)top$(topk).pdf")
    # Plots.savefig(p2, "output/inferredGluExp$(iExp)top$(topk).pdf")
    # Plots.savefig(p3, "output/inferredLacExp$(iExp)top$(topk).pdf")
end


myplots = []
legends = [:topleft :bottomleft :topleft]
mycolors = [:blue :red :green :purple]
ylabels = [L"Cell density $(1/m^2)$" L"Glucose concentration $(mmol/l)$" L"Lactate concentration $(mmol/l)$"]
thisplot = Plots.plot(layout=(nExp, 3), size=(2000, 1200))
for iExp in 1:nExp
  plotRange = (iExp - 1) * numberOfReplicates + 1 : iExp * numberOfReplicates
  subplot = (iExp - 1) * 3
  for iState in 1:length(ylabels)
    p = Plots.plot!(htstepsinferred,[inferredsim[i][iState,:] for i in plotRange], lw=3, color = mycolors, subplot = subplot + iState,
    label = plotlabelsSilico, xlabel = L"Time $(h)$", ylabel = ylabels[iState], title = "Exp$(iExp), $(plotTitles[iExp])")
    Plots.scatter!(p, htstepsinferred, dataDict["observations"][iState,:,plotRange],  yerr = coalesce.(dataDict["stds"][iState,:,plotRange], 0), 
    subplot = subplot + iState, color = mycolors, msc = mycolors, legend = legends[iState], label = plotlabelsVivo)
    # append!(myplots, p)
  end
end


numberOfReplicates = 3
legends = [:topleft :bottomleft :topleft]
legends_makie = [:lt :lb :lt]
mycolors = [:blue :red :green :purple]
ylabels = [L"Cell density $(1/m^2)$" L"Glucose concentration $(mmol/l)$" L"Lactate concentration $(mmol/l)$"]

# thisplot = CairoMakie.Figure(layout=(nExp, length(ylabels)), resolution=(2500, 1800))
thisplot = CairoMakie.Figure(resolution=(2000, 1400))
myaxes = [CairoMakie.Axis(thisplot[i, j]) for j in 1:length(ylabels), i in 1:nExp]

p, pexp = [], []
for iExp in 1:nExp
  plotRange = (iExp - 1) * numberOfReplicates + 1 : iExp * numberOfReplicates
  subplot = (iExp - 1) * 3

  for iState in eachindex(ylabels)
    this_axes = myaxes[subplot + iState]
    this_axes.title = "Experiment $(iExp), $(plotTitles[iExp])"
    this_axes.ylabel = ylabels[iState]
    this_axes.xlabel = L"Time $(h)$"
    ydata = [inferredsim[i][iState,:] for i in plotRange]
    p = [CairoMakie.lines!(this_axes, htstepsinferred, ydata[idata], color = mycolors[idata]) 
      for idata in eachindex(ydata)]

    pexp = [CairoMakie.scatter!(this_axes, htstepsinferred, 
      coalesce.(dataDict["observations"][iState,:,plotRange][:, idata], NaN),  
      color = mycolors[idata], msc = mycolors[idata])
      for idata in eachindex(ydata)]

    perrors = [CairoMakie.errorbars!(this_axes, htstepsinferred, 
      coalesce.(dataDict["observations"][iState,:,plotRange][:, idata], NaN),
      coalesce.(dataDict["stds"][iState,:,plotRange][:, idata], 0), 
      color = mycolors[idata], msc = mycolors[idata], whiskerwidth = 10)
      for idata in eachindex(ydata)]
    
  end
end
Legend(thisplot[nExp + 1, begin:length(ylabels)], [p..., pexp...], [plotlabelsSilico..., plotlabelsVivo...], 
        nbanks = 4)
rowsize!(thisplot.layout, 5, Relative(1/20))

display(thisplot)

save(datadir("sims", "figures", "InferredModel.pdf"), thisplot)

mycolors = [:blue :red :green :purple]
ylabelsErrors = ["Cell density (%)" "Glucose concentration (%)" "Lactate concentration (%)"]

thisplot = CairoMakie.Figure(resolution=(2000, 1400))
myaxes = [CairoMakie.Axis(thisplot[i, j]) for j in 1:length(ylabelsErrors), i in 1:nExp]

perrors, pexp = [], []
for iExp in 1:nExp
  plotRange = (iExp - 1) * numberOfReplicates + 1 : iExp * numberOfReplicates
  subplot = (iExp - 1) * 3

  for iState in eachindex(ylabelsErrors)
    this_axes = myaxes[subplot + iState]
    this_axes.title = "Experiment $(iExp), $(plotTitles[iExp])"
    this_axes.ylabel = ylabelsErrors[iState]
    this_axes.xlabel = L"Time $(h)$"
    this_axes.limits = (nothing, [-100 100])
    ydata = [inferredsim[i][iState,:] for i in plotRange]

    perrors = [CairoMakie.scatter!(this_axes, htstepsinferred, 
      (ydata[idata] .- coalesce.(dataDict["observations"][iState,:,plotRange][:, idata], NaN)) ./
      coalesce.(dataDict["observations"][iState,:,plotRange][:, idata], NaN) .* 100,  
      color = mycolors[idata], msc = mycolors[idata], markersize = 20)
      for idata in eachindex(ydata)]

    pexp = [CairoMakie.errorbars!(this_axes, htstepsinferred, 
      0 .* coalesce.(dataDict["observations"][iState,:,plotRange][:, idata], NaN),
      coalesce.(dataDict["stds"][iState,:,plotRange][:, idata], 0) ./ 
      coalesce.(dataDict["observations"][iState,:,plotRange][:, idata], NaN) .* 100, 
      color = mycolors[idata], msc = mycolors[idata], whiskerwidth = 10)
      for idata in eachindex(ydata)]
    
  end
end

plotlabelsErrors = ["Model error, 25 000 initial cells" "Model error, 50 000 initial cells" "Model error, 100 000 initial cells" "Model error, 200 000 initial cells"]
plotlabelsNoises = ["Experiment noise, 25 000 initial cells" "Experiment noise, 50 000 initial cells" "Experiment noise, 100 000 initial cells" "Experiment noise, 200 000 initial cells"]

Legend(thisplot[nExp + 1, begin:length(ylabelsErrors)], [perrors..., pexp...], [plotlabelsErrors..., plotlabelsNoises...], 
        nbanks = 4)
rowsize!(thisplot.layout, 5, Relative(1/20))

display(thisplot)
save(datadir("sims", "figures", "InferredErrors.pdf"), thisplot)

allerrors, allnoises = [], []

for iExp in 1:nExp
  plotRange = (iExp - 1) * numberOfReplicates + 1 : iExp * numberOfReplicates

  for iState in eachindex(ylabelsErrors)
    ydata = [inferredsim[i][iState,:] for i in plotRange]

    these_errors = [
      (ydata[idata] .- dataDict["observations"][iState,:,plotRange][:, idata]) ./
      dataDict["observations"][iState,:,plotRange][:, idata]
      for idata in eachindex(ydata)]

    these_noises = [
      dataDict["stds"][iState,:,plotRange][:, idata] ./ 
      dataDict["observations"][iState,:,plotRange][:, idata]
      for idata in eachindex(ydata)]
    append!(allerrors, these_errors)
    append!(allnoises, these_noises)
    
  end
end

allerrors = coalesce.(permutedims(hcat(allerrors...)), 0)
allnoises = coalesce.(permutedims(hcat(allnoises...)), 0)

mean_error = sum(abs.(allerrors)) / sum(allerrors .!= 0)
mean_noise = sum(allnoises) / sum(allnoises .!= 0)
