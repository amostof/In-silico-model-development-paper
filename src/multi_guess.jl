"""
For generating performing optimization on multiple datasets. The loss function 
is provided in 'loss', the initial guess for optimizer is 'initialp' and the 
synthetic datasets are provided in 'noisydataAll'.

"""
function diffData(loss, initialp, noisydataAll, prob, prob_func; maxiters = maxiters, alg = :PolyOpt, BFGS = true, kwargs...)
    niter = length(noisydataAll)
    params = Array{Vector{Float64},1}(undef, niter)
    errors = Array{Float64,1}(undef, niter)
    losses = Array{Float64,1}(undef, niter)

    for (iter, noisydata) in enumerate(noisydataAll)
        if alg == :PolyOpt
        result_ode = PolyOptModified(loss, noisydata, initialp, callback, maxiters, prob, prob_func; 
                                    BFGSsolver = BFGS, kwargs...)
        elseif alg == :MultiStart
        result_ode = MultistartAdam(loss, noisydata, initialp, callback, maxiters, prob, prob_func; 
                                    tiktak_iters = 100, lb = 1E-6, ub = 1E+8, kwargs...)
        end

        finalp = exp.(result_ode.u)
        params[iter] = finalp
        errors[iter] = norm(finalp - p) ./ norm(p) .* 100
        losses[iter] = result_ode.minimum
    end
    return params, errors, losses
end

"""
This function performs optimization using multiple initial parameter guesses 
provided by 'initialps'. This function can also perform optimization on multiple
datasets by calling 'diffData'. Loss function is provided by 'loss'.
"""
function diffInitialGuess(loss, initialps, noisydataAll, prob, prob_func; kwargs...)
    nguess = size(initialps, 2)
    paramsAll = Array{Any,1}(undef, nguess) 
    errorsAll = Array{Any,1}(undef, nguess)
    lossesAll = Array{Any,1}(undef, nguess)
    
    println("Real parameter, $(round.(p, sigdigits=3))")
    for iguess in 1:nguess
      println("Initial guess $(iguess), $(round.(initialps[:, iguess], sigdigits=3))")
      params, errors, losses = diffData(loss, initialps[:, iguess], noisydataAll, prob, prob_func; kwargs...)
  
      errorsAll[iguess] = errors
      paramsAll[iguess] = params
      lossesAll[iguess] = losses
    end
    return paramsAll, errorsAll, lossesAll
end

"""
This function is equivalent to 'diffInitialGuess' but is for when only one dataset exists
and we are working with real data.

"""
function diffInitialGuessData(loss, initialps, dataDict, prob, prob_func; alg = :PolyOpt, BFGS = true)
    nguess = size(initialps, 2)
    paramsAll = Array{Any,1}(undef, nguess); 
    plotlossAll = Array{Any,1}(undef, nguess); 
    lossesAll = Array{Any,1}(undef, nguess)
    # println("Real parameter, $(round.(p, sigdigits=3))")
    # Threads.@threads 
    for iguess in 1:nguess
      println("Initial guess $(iguess), $(round.(initialps[:, iguess], sigdigits=3))")
  
      plotloss = []
  
      if alg == :PolyOpt
        result_ode = PolyOptModified(loss, dataDict, initialps[:, iguess], callback, maxiters, prob, prob_func; BFGSsolver = BFGS)
    
      elseif alg == :MultiStart
        result_ode = MultistartAdam(loss, dataDict, initialps[:, iguess], callback, maxiters, prob, prob_func;
                                tiktak_iters = 100, lb = 1E-12, ub = 1E+12)
      end
  
      plotlossAll[iguess] = plotloss
      paramsAll[iguess] = [exp.(result_ode.u)]
      lossesAll[iguess] = result_ode.minimum
    end
    return paramsAll, plotlossAll, lossesAll
  end