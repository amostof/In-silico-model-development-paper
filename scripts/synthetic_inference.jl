niter = 5
noisydataAll = noiseGenerator(noiselevel, sim; niter = niter);
params, errors, losses = diffData(loss, initialp, noisydataAll, prob_func; alg = :MultiStart, save_idxs=save_idxs, useLikelihood = false);

# display(histogram(errors))
# display(histogram(losses))

# display(marginalhist(permutedims(hcat(params...))[:, 1], permutedims(hcat(params...))[:, 2], fc=:plasma))
noinflosses = losses[.~ isinf.(losses)]; noinferrors = errors[.~ isinf.(losses)]; noinfparams = params[.~ isinf.(losses)]
println("\n************************************* \n Analysis complete. ")
println("Ground truth: $(p)\nInitial Guess: $(initialp)\nNoise level: $(noiselevel*100)%")
println("tsteps: $(ddt/3600)h")
println("Final parameters: $(round.(mean(noinfparams), sigdigits=5)), std: $(round.(std(noinfparams), sigdigits=5))")
println("Error: $(round(mean(noinferrors), sigdigits=3))%, std: $(round(std(noinferrors), sigdigits=3))%")
println("Loss: $(round(mean(noinflosses), sigdigits=3)), std: $(round(std(noinflosses), sigdigits=3))")
println("Error per parameter: $(round.((mean(noinfparams) - p) ./ (p) .* 100, sigdigits=3))%")

searchgrid = [1E-9, 1E-1]
bounds = fill(searchgrid,length(p)).*p

nguess = 100
initialps = latinCube(bounds, length(p), nguess)
paramsAll, errorsAll, lossesAll = diffInitialGuess(loss, initialps, noisydataAll, prob_func; save_idxs=save_idxs, useLikelihood = false, BFGS = false, alg = :PolyOpt);

paramsAllhcat =  permutedims(hcat(paramsAll...))
errorsAllhcat =  permutedims(hcat(errorsAll...))
lossesAllhcat =  permutedims(hcat(lossesAll...))

# noinflosses = losses[.~ isinf.(losses)]; noinferrors = errors[.~ isinf.(losses)]; noinfparams = params[.~ isinf.(losses)]

bestGuesses = [argmin(lossesAllhcat, dims = 1)[i][1] for i in 1:size(lossesAllhcat)[2]]
corrErrors = [errorsAllhcat[bestGuesses[i],i] for i in 1:size(errorsAllhcat)[2]]
bestLosses = [minimum(lossesAllhcat, dims = 1)[i][1] for i in 1:size(lossesAllhcat)[2]]
corrParams = [paramsAllhcat[bestGuesses[i],i] for i in 1:size(paramsAllhcat)[2]]

lossAtRealParameters = []
for i in eachindex(noisydataAll)
  append!(lossAtRealParameters, loss(p, noisydataAll[i]; 
  logscale = false, save_idxs = save_idxs, onlyLoss = true, useLikelihood = false))
end

lossAtRealParameters = convert(Array{Float64,1}, lossAtRealParameters)

println("\n************************************* \n Analysis complete. ")
println("Ground truth: $(p)\nNumber of parameter guesses: $(nguess)\nSearch Grid: $(searchgrid)\nNoise level: $(noiselevel*100)%")
println("Final parameters: $(round.(mean(corrParams), sigdigits=5)), std: $(round.(std(corrParams), sigdigits=5))")
println("Error: $(round(mean(corrErrors), sigdigits=3))%, std: $(round(std(corrErrors), sigdigits=3))%")
println("Loss: $(round(mean(bestLosses), sigdigits=3)), std: $(round(std(bestLosses), sigdigits=3))")
println("Loss at Real Parameters: $(round.(mean(lossAtRealParameters), sigdigits=3)), std: $(round.(std(lossAtRealParameters), sigdigits=3))")




loglikelihoodAtInferredParameters = []
for i in eachindex(noisydataAll)
  append!(loglikelihoodAtInferredParameters, loss(p, noisydataAll[i]; 
  logscale = false, save_idxs = save_idxs, onlyLoss = true, useLikelihood = true))
end


temp_dict = Dict{String, Any}("p" => p, 
  "nguess" => nguess, 
  "searchgrid" =>searchgrid,
  "noiselevel" => noiselevel,
  "noisydataAll" => noisydataAll,
  "initialps" => initialps,
  "paramsAllhcat" => paramsAllhcat,
  "errorsAllhcat" => errorsAllhcat,
  "lossesAllhcat" => lossesAllhcat,
  "corrParams" => corrParams,
  "corrErrors" => corrErrors,
  "bestLosses" => bestLosses,
  "lossAtRealParameters" => lossAtRealParameters,
  "nps" => length(p),
  "nu0s" => size(u0s,1));

@tagsave(datadir("sims", savename(temp_dict, "jld2")), temp_dict)



experimentalDesignData = CSV.read(datadir("sims", "ExperimentalDesignData.csv"), DataFrame)
experimentalDesignPlot = Plots.plot(
  experimentalDesignData[!, "Sampling period (h)"], 
  experimentalDesignData[!, "Error (%)"],
  xlabel = L"Sampling period $(h)$",
  ylabel = L"Error $(\%)$",
  linewidth = 3,
  size = (500, 300),
  ylims = (0, 45),
  xlims = (0, 49),
  legend = false)

savefig(experimentalDesignPlot, datadir("sims", "figures", "ExperimentalDesign.pdf"))

experimentalDesignPlot
experimentalDesignData