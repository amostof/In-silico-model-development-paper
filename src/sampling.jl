function latinCube(bounds, dims, nguess = 100)  
    initialps = []
    plan, _ = LHCoptim(nguess, dims, 1000)
    plan /= nguess
    
    for i in 1:dims
        append!(initialps, [quantile(LogUniform(bounds[i][1], bounds[i][2]), plan[:,i])])
    end
    return permutedims(hcat(initialps...))
end