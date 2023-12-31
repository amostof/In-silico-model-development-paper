function dedimension(p)
    return [ustrip(p_i) for p_i in p]
end

function prob_func(prob,i,repeat)
    remake(prob,u0=u0s[i,:])
end

function prob_funcData(prob,i,repeat)
    remake(prob, u0=dataDict["u0"][i,:])
end

median_interval(d,p = 0.95) = [quantile(d,1-(1+p)/2), quantile(d,(1+p)/2)]

# median_interval(d; dims = 1, p = 0.95) = quantile(d,1-(1+p)/2), quantile(d,(1+p)/2)