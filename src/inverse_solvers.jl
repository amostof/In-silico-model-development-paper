""" I implemented PolyOpt() from OptimizationPolyalgorithms to check
the effect of different Flux.jl optimizers. 

Input `initialp` is the initial guesses for parameters in real scale while the
output contains `result_ode` with `result_ode.u` in log-scale. So you need to
do exp.(result_ode.u) before using the inferred parameters.

I learned that using the the same learn rate and all other parameters 
set to default, the following solvers return the exact same output.
- Adam
- AdaMax
- AdamW
- NAdam
- OAdam

And the rest of the optimizer do not work, namely,
- RAdam, AMSGrad, AdaBelief, AdaDelta, AdaGrad, RMSProp, Nesterov, Momentum

For reference, look at [Optimisers.jl API](https://fluxml.ai/Optimisers.jl/dev/api/).
"""
function PolyOptModified(loss, noisydata, initialp, callback, maxiters, prob, prob_func; 
    opt = Optimisers.Adam(0.01), bounded = false, BFGSsolver = true, kwargs...)

    # AutoForwardDiff is faster as it is forward AD (Zygote is backward AD),
    # ain cases with less than 100 ODEs.

    adtype = Optimization.AutoForwardDiff()
    lb = nothing
    ub = nothing

    if bounded
        lb = log.(initialp .* 1E-10)
        ub = log.(initialp .* 1E+10)
    end

    optf = Optimization.OptimizationFunction((x, p) -> loss(x, noisydata, prob, prob_func; kwargs...), adtype)
    optprob = Optimization.OptimizationProblem(optf, log.(initialp), lb = lb, ub = ub)
    result_ode = Optimization.solve(optprob, opt, callback = callback, maxiters = maxiters)

    if BFGSsolver
        optprob2 = Optimization.OptimizationProblem(optf, result_ode.u)
        result_ode = Optimization.solve(optprob2, BFGS(initial_stepnorm = 0.01), callback = callback, maxiters = maxiters)
    end
    return result_ode

end

"""  
"""
function MultistartAdam(loss, noisydata, initialp, callback, maxiters, prob, prob_func; 
    tiktak_iters = 200, lb = 1E-5, ub = 1E+5, kwargs...)

    adtype = Optimization.AutoForwardDiff()
    opt = LBFGS() # Optimisers.Adam(0.01)
    optf = Optimization.OptimizationFunction((x, p) -> loss(x, noisydata, prob, prob_func; kwargs...), adtype)
    optprobtik = Optimization.OptimizationProblem(optf, log.(initialp), lb = log.(initialp .* lb), ub = log.(initialp .* ub))

    result_ode = solve(optprobtik, MultistartOptimization.TikTak(tiktak_iters), opt, maxiters = maxiters, callback = callback)
    return result_ode

end