numberOfEquations = 4

oxygen = dedimension([co₀[1] co₀[1] co₀[2] co₀[2]])

nExp = 4
rname = 1
dataArrayDict = Vector{Any}(undef, nExp)

for iExp in 1:nExp

    # Read the files
    inputCell = CSV.read(joinpath(@__DIR__, "..", "data", "exp_pro", "BEAS2BsExp1234", "Exp$(iExp)ComsolCellPopAll.csv"), DataFrame)
    inputConcs = CSV.read(joinpath(@__DIR__, "..", "data", "exp_pro", "BEAS2BsExp1234", "Exp$(iExp)ComsolConcsAll.csv"), DataFrame)


    # Write everything to the dictionary
    dataArrayDict[iExp] = Dict{String, Any}()
    dataArrayDict[iExp] = readDataPerExperiment(inputCell, inputConcs, oxygen[iExp])
end

dataDict = combineArrayDictionary(dataArrayDict)

# Hacky way to convert populations to densities
dataDict["u0"][:,1] = map(n0 -> ustrip(uconvert(u"mol/m^3", n0 ./ wellArea ./ celldia .* 1u"mol")), dataDict["u0"][:,1])
dataDict["observations"][1,:,:] = map(obs -> ustrip(uconvert(u"mol/m^3", obs ./ wellArea ./ celldia .* 1u"mol")), dataDict["observations"][1,:,:])
dataDict["stds"][1,:,:] = map(obs -> ustrip(uconvert(u"mol/m^3", obs ./ wellArea ./ celldia .* 1u"mol")), dataDict["stds"][1,:,:])

trainingSize = 0.6
validationSize = 0.2

trainingDict, validationDict, testingDict = threewaySplit(dataDict; sampler = MersenneTwister(1234), trainingSize, validationSize);
