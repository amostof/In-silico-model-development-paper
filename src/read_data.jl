function readDataPerExperiment(inputCell, inputConcs, oxygen)

    # Start by reading initial conditions
    numberOfInitialPops = length(unique(inputCell."initialCells"))
    inputInitialConditions = [unique(inputCell."initialCells"), repeat(unique(inputCell."initialGlucose (mol/m^3)"), numberOfInitialPops),
                                            repeat(unique(inputCell."initialLactate (mol/m^3)"), numberOfInitialPops)]
    inputInitialConditions = hcat(inputInitialConditions...)
    inputInitialConditions = hcat(inputInitialConditions, repeat([oxygen], numberOfInitialPops))

    allInitialConditions = Array{Float64}(undef, 0, numberOfEquations)
    allInitialConditions = vcat(allInitialConditions, inputInitialConditions)
    
    # Then read measurment data
    inputCellGrouped = groupby(inputCell, ["initialCells", "t (s)"])
    inputConcGrouped = groupby(inputConcs, ["initialCells", "t (s)"])

    meanCellGrouped = combine(inputCellGrouped, "experimentalCells (1)" => mean)
    stdCellGrouped = combine(inputCellGrouped, "experimentalCells (1)" => std)

    meanConcGrouped = combine(inputConcGrouped, ["experimentalGlucose (mol/m^3)", "experimentalLactate (mol/m^3)"] .=> mean)
    stdConcGrouped = combine(inputConcGrouped, ["experimentalGlucose (mol/m^3)", "experimentalLactate (mol/m^3)"] .=> std)


    # Convert data to means and standard deviations
    stdCell = unstack(stdCellGrouped, "initialCells", "experimentalCells (1)_std")
    meanCell = unstack(meanCellGrouped, "initialCells", "experimentalCells (1)_mean")

    stdLac = unstack(stdConcGrouped[:, Not("experimentalGlucose (mol/m^3)_std")], "initialCells", "experimentalLactate (mol/m^3)_std")
    meanLac = unstack(meanConcGrouped[:, Not("experimentalGlucose (mol/m^3)_mean")], "initialCells", "experimentalLactate (mol/m^3)_mean")

    stdGlu = unstack(stdConcGrouped[:, Not("experimentalLactate (mol/m^3)_std")], "initialCells", "experimentalGlucose (mol/m^3)_std")
    meanGlu = unstack(meanConcGrouped[:, Not("experimentalLactate (mol/m^3)_mean")], "initialCells", "experimentalGlucose (mol/m^3)_mean")

    # Reformat the data to the format needed by EnsembleProblem
    dataPerExperiment = Array{Float64}(undef, numberOfEquations, size(meanGlu, 1), 0)
    stdPerExperiment = Array{Float64}(undef, numberOfEquations, size(meanGlu, 1), 0)

    for i in 1:numberOfInitialPops
        dataPerInitialCell = Array(transpose(hcat(meanCell[:, i+1], meanGlu[:, i+1], meanLac[:, i+1], repeat([oxygen], size(meanGlu, 1)))))
        stdPerInitialCell = Array(transpose(hcat(stdCell[:, i+1], stdGlu[:, i+1], stdLac[:, i+1], repeat([1], size(stdGlu, 1)))))
        dataPerExperiment = cat(dataPerExperiment, dataPerInitialCell, dims = 3)
        stdPerExperiment = cat(stdPerExperiment, stdPerInitialCell, dims = 3)
    end
    
    return Dict("u0" => allInitialConditions, "observations" => dataPerExperiment, 
                    "stds" => stdPerExperiment, "tsteps" => unique(inputCell."t (s)"))
    
end

function combineArrayDictionary(dataArrayDict)

  # Union over all time steps
  timeSet = Set()
  for iExp in 1:length(dataArrayDict)
      union!(timeSet, dataArrayDict[iExp]["tsteps"])
  end
  allTimes = sort(collect(timeSet))

  numberOfEquations = size(dataArrayDict[1]["u0"], 2)
  numberOfInitialPops = size(dataArrayDict[1]["u0"], 1)

  # Create the output
  dataAll = Array{Float64}(undef, numberOfEquations, length(allTimes), 0)
  stdsAll = Array{Float64}(undef, numberOfEquations, length(allTimes), 0)
  uInitial = Array{Float64}(undef, 0, numberOfEquations)
  for iExp in 1:length(dataArrayDict)
      numberOfInitialPops = 4
      dataMatrix = Array{Float64}(undef, numberOfEquations, 0, numberOfInitialPops)
      stdsMatrix = Array{Float64}(undef, numberOfEquations, 0, numberOfInitialPops)
      jTime = 1
      for iTime in 1:length(allTimes)
          if allTimes[iTime] == dataArrayDict[iExp]["tsteps"][jTime]
              dataMatrix = cat(dataMatrix, reshape(dataArrayDict[iExp]["observations"][:, jTime, :], 
                              (numberOfEquations, 1, numberOfInitialPops)), dims = 2)
              stdsMatrix = cat(stdsMatrix, reshape(dataArrayDict[iExp]["stds"][:, jTime, :], 
                              (numberOfEquations, 1, numberOfInitialPops)), dims = 2)
              
              if jTime != length(dataArrayDict[iExp]["tsteps"])
                  jTime += 1
              end
          else
              dataMatrix = cat(dataMatrix, reshape(repeat([missing], numberOfEquations * numberOfInitialPops), 
                              (numberOfEquations, 1, numberOfInitialPops)), dims = 2)
              stdsMatrix = cat(stdsMatrix, reshape(repeat([missing], numberOfEquations * numberOfInitialPops), 
                              (numberOfEquations, 1, numberOfInitialPops)), dims = 2)
          end
      end
      dataAll = cat(dataAll, dataMatrix, dims = 3)
      stdsAll = cat(stdsAll, stdsMatrix, dims = 3)
      uInitial = cat(uInitial, dataArrayDict[iExp]["u0"], dims = 1)
  end

  allTimes = allTimes[1:end-1]
  dataAll = dataAll[:,1:end-1,:]
  stdsAll = stdsAll[:,1:end-1,:]

  # uInitial = uInitial[1:4,:]
  # dataAll = dataAll[:,:,1:4]
  # stdsAll = stdsAll[:,:,1:4]

  return Dict("u0" => uInitial, "observations" => dataAll, 
              "stds" => stdsAll, "tsteps" => allTimes)
end