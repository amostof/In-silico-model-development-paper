function forwardSolver(p, tsteps, u0s, prob, prob_func; save_idxs=nothing, kwargs...)
    _prob = remake(prob, p = p)
    _ensemble_prob = EnsembleProblem(_prob, prob_func = prob_func)
    pred = solve(
        _ensemble_prob,
        Rosenbrock23(autodiff = false),
        EnsembleThreads(),
        maxiters = Int(1e6),
        trajectories = size(u0s)[1],
        saveat = tsteps,
        save_idxs = save_idxs,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
        kwargs...
    )
    return pred
end