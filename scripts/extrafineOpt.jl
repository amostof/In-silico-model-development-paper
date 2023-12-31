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

thissims = ["mmkPconstmmkN", "mmkPmmkPmmkN", "constconstmmkN", "mmkPmmkPconst", "constmmkPmmkN"]

extrafineModels = permutedims([:const :const :mmkN
                                :mmkP :const :mmkN
                                :mmkP :mmkP :const
                                :mmkP :mmkP :mmkN
                                :const :mmkP :mmkN])

function extrafineOpt(alldata, thissims, trainingDict; topk = 3, BFGS = false)
    
    sims = alldata["simulations"]
    extrafineDict =  Dict{String, Any}("topk" => topk,
        "simulations" => Dict{String, Any}())

    for (simkey, sim) in sims 
        if simkey in thissims
            perms = sortperm(vec(sim["lossesAllhcat"]))[begin:topk]
            newps = hcat(sim["paramsAllhcat"][perms]...)
            
            modelTypes = sim["modelTypes"]
            probCase = ODEProblem(neotissue3EQsCases!, u0, tspan, p)

            println("simkey = ", simkey)
            paramsAll, _, lossesAll = diffInitialGuessData(lossRealData, newps, trainingDict, probCase, prob_funcData; alg = :PolyOpt, BFGS = BFGS);

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
            println("Ground truth: $(p)\nNumber of parameter guesses: $(topk)")
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
            
            extrafineDict["simulations"][string(string.(modelTypes)...)] = temp_dict
        end
    end
    return extrafineDict
end

extrafineDict = extrafineOpt(alldata, thissims, trainingDict; topk = 200)
extrafineBFGSDict = extrafineOpt(alldata, thissims, trainingDict; BFGS = true, topk = 15)

output_dictfine = Dict{String, Any}(
"alldata" => alldata,
"extrafineDict" => extrafineDict, 
"extrafineBFGSDict" => extrafineBFGSDict)

@tagsave(datadir("sims", "Exp", "Finerdata.jld2"), output_dictfine)


for simkey in thissims
    println("#### $simkey validation #################################")
    print("Alldata negative double log likelihood: ")
    println("$(alldata["simulations"][simkey]["validLossOverNo"] * sum(validationDict["observations"] .!== missing) * nObeservationsPerPoint)")
    print("extrafineDict negative double log likelihood: ")
    println("$(extrafineDict["simulations"][simkey]["validLossOverNo"] * sum(validationDict["observations"] .!== missing) * nObeservationsPerPoint)")
    print("extrafineBFGSDict negative double log likelihood: ")
    println("$(extrafineBFGSDict["simulations"][simkey]["validLossOverNo"] * sum(validationDict["observations"] .!== missing) * nObeservationsPerPoint)")
    println("")
end


for simkey in thissims
    println("#### $simkey calibration #################################")
    print("Alldata negative double log likelihood: ")
    println("$(alldata["simulations"][simkey]["bestLosses"][1] * nObeservationsPerPoint)")
    print("extrafineDict negative double log likelihood: ")
    println("$(extrafineDict["simulations"][simkey]["bestLosses"][1] * nObeservationsPerPoint)")
    print("extrafineBFGSDict negative double log likelihood: ")
    println("$(extrafineBFGSDict["simulations"][simkey]["bestLosses"][1] * nObeservationsPerPoint)")
    println("")

end

for simkey in thissims
    println("#### $simkey calibration #################################")
    print("Alldata: ")
    println("$(round.(alldata["simulations"][simkey]["corrParams"][1], sigdigits = 4))")
    print("extrafineDict: ")
    println("$(round.(extrafineDict["simulations"][simkey]["corrParams"][1], sigdigits = 4))")
    print("extrafineBFGSDict: ")
    println("$(round.(extrafineBFGSDict["simulations"][simkey]["corrParams"][1], sigdigits = 4))")
    println("")

end