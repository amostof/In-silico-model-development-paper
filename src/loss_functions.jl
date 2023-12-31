
ForwardDiff.can_dual(::Type{Union{Missing, Float64}}) = true

function loss(p, observations, prob, prob_func; logscale = true, onlyLoss = false, 
            useJacobian = false, useLikelihood = true, squaring = true, kwargs...)
    if logscale == true
        temp_p = exp.(p)
    else
        temp_p = p
    end

    pred = forwardSolver(temp_p, tsteps, u0s, prob, prob_func; kwargs...)

    if any((s.retcode != :Success for s in pred))
        return Inf, pred, observations
    end

    if useLikelihood
        normalizer = pred * noiselevel # Divide everything by the standard deviation
    else
        normalizer = mean(observations, dims = 2) # Divide everything by the average value
    end

    norm_pred = pred ./ normalizer
    norm_obs  = observations ./ normalizer

    resized_pred = [[norm_pred[:, j, i] for j in axes(norm_pred, 2)]
                    for i in axes(norm_pred, 3)]
    resized_obs = [[norm_obs[:, j, i] for j in axes(norm_obs, 2)]
                   for i in axes(norm_obs, 3)]

    if useJacobian
        Jst = [map(u -> I +
                        0.1 *
                        ForwardDiff.jacobian(UJacobianWrapper(neotissue3EQs!, 0.0, temp_p),
                                             u), resized_pred[i]) for i in 1:length(sim)]
        diff = [map((J, u, data) -> J * (abs2.(u .- data)), Jst[i], resized_pred[i],
                    resized_obs[i]) for i in axes(resized_obs, 1)]
    else
        diff = [map((u, data) -> abs2.(u .- data), resized_pred[i], resized_obs[i])
                for i in axes(resized_obs, 1)]
    end

    loss = sum(sum(sum([map(data -> abs.(data), diff[i]) for i in axes(diff, 1)])))

    if squaring && !useLikelihood
        loss = sqrt(loss)
    end

    if onlyLoss
        return loss
    end
    return loss, pred, observations
end

function lossRealData(p, dataDict, prob, prob_func; logscale = true, onlyLoss = false, 
                useJacobian = false, useLikelihood = true, squaring = true, kwargs...)
    if logscale == true
        temp_p = exp.(p)
    else
        temp_p = p
    end

    tsteps = dataDict["tsteps"]
    u0s = dataDict["u0"]
    observations = dataDict["observations"]

    pred = forwardSolver(temp_p, tsteps, u0s, prob, prob_func; kwargs...)

    if any((s.retcode != :Success for s in pred))
        return Inf, pred, observations
    end

    if useLikelihood
        normalizer = coalesce.(dataDict["stds"], 1) # Divide everything by the standard deviation
    else
        normalizer = mean(coalesce.(observations, 0), dims = 2) # Divide everything by the average value
    end
    
    norm_pred = pred ./ normalizer
    norm_obs  = observations ./ normalizer

    resized_pred = [[norm_pred[:, j, i] for j in axes(norm_pred, 2)]
                    for i in axes(norm_pred, 3)]
    resized_obs = [[norm_obs[:, j, i] for j in axes(norm_obs, 2)]
                   for i in axes(norm_obs, 3)]

    if useJacobian
        Jst = [map(u -> I +
                        0.1 *
                        ForwardDiff.jacobian(UJacobianWrapper(neotissue3EQs!, 0.0, temp_p),
                                             u), resized_pred[i])
               for i in 1:length(resized_pred)]
        diff = [map((J, u, data) -> J * (abs2.(u .- data)), Jst[i], resized_pred[i],
                    resized_obs[i]) for i in axes(resized_obs, 1)]
    else
        diff = [map((u, data) -> abs2.(u .- data), resized_pred[i], resized_obs[i])
                for i in axes(resized_obs, 1)]
    end
    loss = sum(sum(sum([map(data -> abs.(coalesce.(data, 0)), diff[i])
                             for i in axes(diff, 1)])))

    if squaring && !useLikelihood
        loss = sqrt(loss)
    end
    
    if onlyLoss
        return loss
    end
    return loss, pred, observations
end