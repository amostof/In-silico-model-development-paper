function threewaySplit(dataDict; sampler = Random.GLOBAL_RNG, trainingSize = 0.6, validationSize = 0.2)
    trainingDict = deepcopy(dataDict)
    validationDict = deepcopy(dataDict)
    testingDict = deepcopy(dataDict)

    for iExp in axes(dataDict["observations"], 3)
        # Find all the non-missing indices 
        allIndices = findall(dataDict["observations"][1, :, iExp] .!== missing)
        noObs = length(allIndices)

        # Select the index of the training data
        sampled = sample(sampler, allIndices, convert(Int, noObs * trainingSize);
            replace=false, ordered=true)

        # Select the index of the validation data
        notSampled = setdiff(allIndices, sampled)
        validSampled = sample(sampler, notSampled, convert(Int, noObs * validationSize);
            replace=false, ordered=true)

        # Select the index of the test data
        testSampled = setdiff(notSampled, validSampled)

        # Create the three data sets
        for iObserve in axes(dataDict["observations"], 1),
            iMeasure in axes(dataDict["observations"], 2)

            # Training
            if iMeasure ∉ sampled
                trainingDict["observations"][iObserve, iMeasure, iExp] = missing
                trainingDict["stds"][iObserve, iMeasure, iExp] = missing
            end

            # Validation
            if iMeasure ∉ validSampled
                validationDict["observations"][iObserve, iMeasure, iExp] = missing
                validationDict["stds"][iObserve, iMeasure, iExp] = missing
            end

            # Testing
            if iMeasure ∉ testSampled
                testingDict["observations"][iObserve, iMeasure, iExp] = missing
                testingDict["stds"][iObserve, iMeasure, iExp] = missing
            end
        end
    end
    return trainingDict, validationDict, testingDict
end