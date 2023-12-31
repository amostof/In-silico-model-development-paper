function noiseGenerator(noiselevel, sol; niter = niter)
    noisydataAll = []
    for _ in 1:niter
      noise_multiplier = 1.0 .+ noiselevel .* randn(rng, size(sol))
  
  
      noisydata = sol .* noise_multiplier
  
      append!(noisydataAll, [noisydata])
    end
    return noisydataAll
end