plotloss = []
callback = function (p, l, pred, noisydata)
  temp_p = exp.(p)
  global plotloss
  append!(plotloss, l)

  if !@isdefined printer
    printer = false
  end

  if !@isdefined plotter
    plotter = false
  end
  
  if printer
    println("Loss: $l")
    println("Parameters: $temp_p")
  end
  if plotter
    htsteps = tsteps / 3600.0
    p1 = Plots.plot(htsteps,[pred[i][1,:] for i in 1:length(sim)],lw=3)
    Plots.scatter!(p1, htsteps, noisydata[1,:,:])
    p2 = Plots.plot(htsteps,[pred[i][2,:] for i in 1:length(sim)],lw=3, legend = false)
    Plots.scatter!(p2, htsteps, noisydata[2,:,:])
    p3 = Plots.plot(htsteps,[pred[i][3,:] for i in 1:length(sim)],lw=3, legend = false)
    Plots.scatter!(p3, htsteps, noisydata[3,:,:])
    display(Plots.plot(p1, p2, p3, layout = layout, label = u0sLabels))
  end
  return l == Inf
end