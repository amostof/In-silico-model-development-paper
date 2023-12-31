using ProfileLikelihood

function lossRealDataDummy(p, data; logscale = false, onlyLoss = true, 
    useJacobian = false, useLikelihood = true, squaring = true, kwargs...)

    dataDict, prob, prob_func = data
    - lossRealData(p, dataDict, prob, prob_func; logscale, onlyLoss, 
    useJacobian, useLikelihood, squaring, kwargs...)
end


function setboundProfLikelihood(θ₀, simulation, k; constrain = true)
    lb = θ₀ / k
    ub = θ₀ * k
    
    for i in 1:3
        if (simulation["modelTypes"][i] in [:linear, :const]) && constrain
            lb[i+3] = θ₀[i+3]
            ub[i+3] = θ₀[i+3]
        end
    end
    return lb, ub
end

thissim = "mmkPconstmmkN"
thissim = "mmkPmmkPmmkN"

# thissim = "constconstconst"
thissims = ["mmkPconstmmkN", "mmkPmmkPmmkN", "constconstmmkN", "mmkPmmkPconst", "constmmkPmmkN"]
thissims = ["mmkPmmkPmmkN", ]

profiles = Dict{String, Any}(
    "simulations" => Dict{String, Any}()
)

for thissim in thissims
    θ₀ = simulations[thissim]["corrParams"][1]
    lb, ub = setboundProfLikelihood(θ₀, simulations[thissim], 10)

    modelTypes = simulations[thissim]["modelTypes"]
    likeliProb = LikelihoodProblem(
        lossRealDataDummy, θ₀; 
        data=(trainingDict, probCases, prob_funcData),
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=lb, ub=ub),
    )

    likeliSol = mle(likeliProb, NLopt.LN_NELDERMEAD()) # Optim.NelderMead()) # NLopt.LN_NELDERMEAD())

    # new_lb, new_ub = setboundProfLikelihood(likeliSol.mle, simulations[thissim], 10)
    # resolutions = 200*ones(Int64, length(θ₀))
    # param_ranges = construct_profile_ranges(likeliSol, new_lb, new_ub, resolutions)

    # likeliProf = ProfileLikelihood.profile(likeliProb, likeliSol; param_ranges, alg=NLopt.LN_NELDERMEAD, parallel=true)
    
    temp_dict = Dict{String, Any}(
        "likeliProb" => likeliProb,
        "likeliSol" => likeliSol,
        # "likeliProf" => likeliProf,
        "lb" => lb,
        "ub" => ub,
        # "new_lb" => new_lb,
        # "new_ub" => new_ub
    )

    profiles["simulations"][thissim] = temp_dict
end


function confIntervals(simulation, trainingDict, prob, prob_func; 
    θ₀ = simulation["corrParams"][1],
    mleSolver = NLopt.LD_MMA(),
    likelihoodSolver = NLopt.LN_NELDERMEAD,
    k = 10, resolution = 200)

    lb, ub = setboundProfLikelihood(θ₀, simulation, k)

    modelTypes = simulation["modelTypes"]
    likeliProb = LikelihoodProblem(
        lossRealDataDummy, θ₀; 
        data=(trainingDict, prob, prob_func),
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=lb, ub=ub),
    )

    likeliSol = mle(likeliProb, mleSolver)

    new_lb, new_ub = setboundProfLikelihood(likeliSol.mle, simulation, k)
    resolutions = resolution*ones(Int64, length(θ₀))
    param_ranges = construct_profile_ranges(likeliSol, new_lb, new_ub, resolutions)

    likeliProf = ProfileLikelihood.profile(likeliProb, likeliSol; 
        param_ranges, alg=likelihoodSolver, parallel=true,
    )
    
    temp_dict = Dict{String, Any}(
        "likeliProb" => likeliProb,
        "likeliSol" => likeliSol,
        "likeliProf" => likeliProf,
        "lb" => lb,
        "ub" => ub,
        "new_lb" => new_lb,
        "new_ub" => new_ub
    )
    return temp_dict
end

temp_dict = confIntervals(extrafineBFGSDict["simulations"][thissim], trainingDict, probCases, prob_funcData;
    θ₀ = vec(selectedParams), mleSolver = NLopt.LN_NELDERMEAD)

for thissim in thissims
    print("$thissim minimum: $(- profiles["simulations"][thissim]["likeliSol"].maximum)")
    println("  mininum: $(simulations[thissim]["bestLosses"][1])")

    print("Paramters: $(round.(profiles["simulations"][thissim]["likeliSol"].mle ,sigdigits = 4))")
    println("         $(round.(simulations[thissim]["corrParams"][1], sigdigits = 4))")
    println("")
end

@tagsave(datadir("sims", "Exp", "profiles.jld2"), temp_dict)

likeliSol = temp_dict["likeliSol"]

[new_lb[i] <= likeliSol.mle[i] <= new_ub[i] for i in 1:length(θ₀)]

using Test
for i in 1:7
    display(@test likeliSol.mle[i] ∈ likeliProf.confidence_intervals[i])
end

likeliProf = temp_dict["likeliProf"]

fig = plot_profiles(likeliProf;
    latex_names=latexplabels,
    show_mles=true,
    shade_ci=true,
    nrow=3,
    ncol=3,
    # true_vals=[λ, K, u₀],
    fig_kwargs=(fontsize=30, resolution=(2109.644f0, 1444.242f0)),
    axis_kwargs=(width=600, height=300))