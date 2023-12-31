# Definition of mmHg
Unitful.register(@__MODULE__);
@unit(mmHg, "mmHg", mmHg, 133.322365u"Pa", false)

# Initial Oxygen values
function incubatorOxygen()
  torontoAirPressure = 767u"mmHg"
  incubatorOxygenPP = [0.186, 0.05]
  KHₒ = 1.26u"μmol/l/mmHg"
  co = KHₒ * torontoAirPressure .* incubatorOxygenPP
  co = [uconvert(u"mol/m^3", co_i) for co_i in co]
  return co
end

# Main parameters
function mainparameters()
  cellvol = 2.5u"pl"
  nmax = 1 / cellvol * 1.0u"mol"
  nmax = uconvert(u"mol/m^3", nmax)
  return ustrip(nmax)
end

function neotissue3EQs!(du, u, p, t)
  n, cg, cl, co = u
  # β, δ, Vg, Vo, Ko, Kg, Kl, cbarg, cbaro = p
  β, δ, Vg, Ko, Kg, Kl, cbarg = p
  # (; β, δ, Vg, Vo, Ko, Kg, Kl, cbarg, cbaro) = p
  # β, δ, Vg, Kl, cbarg, = p

  # β, Vg, Vo = p
  nmax = mainparameters()

  du[1] = dn = co / (co + Ko) * cg / (cg + Kg) * Kl / (cl + Kl) * β * n * (1 - n / nmax) - δ * n
  du[2] = dcg = -Vg * n * cg / (cg + cbarg)
  du[3] = dcl = -2 * (-Vg * n * cg / (cg + cbarg)) + 1 / 3 * (-Vo * n * co / (co + cbaro))
  du[4] = dco = 0
end

function neotissue3EQsCases!(du, u, p, t)
  n, cg, cl, co = u
  β, δ, Vg, Ko, Kg, Kl, cbarg = p
  otype, gtype, ltype = modelTypes

  nmax = mainparameters()

  fo = reaction(co; type=otype, k=Ko, c0 = u0s[8, 2])
  fg = reaction(cg; type=gtype, k=Kg, c0 = u0s[8, 3])
  fl = reaction(cl; type=ltype, k=Kl, c0 = u0s[8, 4])

  du[1] = dn = fo * fg * fl * β * n * (1 - n / nmax) - δ * n
  du[2] = dcg = -Vg * n * cg / (cg + cbarg)
  du[3] = dcl = -2 * (-Vg * n * cg / (cg + cbarg)) + 1 / 3 * (-Vo * n * co / (co + cbaro))
  du[4] = dco = 0
end

function reaction(c; type=:const, k=1, c0 = 1)
  if type == :const
    return 1
  elseif type == :linear
    return c / c0
  elseif type == :mmkP
    return c / (c + k)
  elseif type == :mmkN
    return k / (c + k)
  end
end


function neotissue3EQsConsO!(du, u, p, t)
  n, cg, cl, co = u
  β, δ, Vg, Ko, Kg, Kl, cbarg = p

  nmax = mainparameters()

  du[1] = dn = 1 * cg / (cg + Kg) * Kl / (cl + Kl) * β * n * (1 - n / nmax) - δ * n
  du[2] = dcg = -Vg * n * cg / (cg + cbarg)
  du[3] = dcl = -2 * (-Vg * n * cg / (cg + cbarg)) + 1 / 3 * (-Vo * n * co / (co + cbaro))
  du[4] = dco = 0
end

function neotissue3EQsConsG!(du, u, p, t)
  n, cg, cl, co = u
  β, δ, Vg, Ko, Kg, Kl, cbarg = p

  nmax = mainparameters()

  du[1] = dn = co / (co + Ko) * 1 * Kl / (cl + Kl) * β * n * (1 - n / nmax) - δ * n
  du[2] = dcg = -Vg * n * cg / (cg + cbarg)
  du[3] = dcl = -2 * (-Vg * n * cg / (cg + cbarg)) + 1 / 3 * (-Vo * n * co / (co + cbaro))
  du[4] = dco = 0
end

function neotissue3EQsConsL!(du, u, p, t)
  n, cg, cl, co = u
  β, δ, Vg, Ko, Kg, Kl, cbarg = p

  nmax = mainparameters()

  du[1] = dn = co / (co + Ko) * cg / (cg + Kg) * 1 * β * n * (1 - n / nmax) - δ * n
  du[2] = dcg = -Vg * n * cg / (cg + cbarg)
  du[3] = dcl = -2 * (-Vg * n * cg / (cg + cbarg)) + 1 / 3 * (-Vo * n * co / (co + cbaro))
  du[4] = dco = 0
end


function neotissue3EQsConsOG!(du, u, p, t)
  n, cg, cl, co = u
  β, δ, Vg, Ko, Kg, Kl, cbarg = p

  nmax = mainparameters()

  du[1] = dn = 1 * 1 * Kl / (cl + Kl) * β * n * (1 - n / nmax) - δ * n
  du[2] = dcg = -Vg * n * cg / (cg + cbarg)
  du[3] = dcl = -2 * (-Vg * n * cg / (cg + cbarg)) + 1 / 3 * (-Vo * n * co / (co + cbaro))
  du[4] = dco = 0
end

function neotissue3EQsConsGL!(du, u, p, t)
  n, cg, cl, co = u
  β, δ, Vg, Ko, Kg, Kl, cbarg = p

  nmax = mainparameters()

  du[1] = dn = co / (co + Ko) * 1 * 1 * β * n * (1 - n / nmax) - δ * n
  du[2] = dcg = -Vg * n * cg / (cg + cbarg)
  du[3] = dcl = -2 * (-Vg * n * cg / (cg + cbarg)) + 1 / 3 * (-Vo * n * co / (co + cbaro))
  du[4] = dco = 0
end

function neotissue3EQsConsOL!(du, u, p, t)
  n, cg, cl, co = u
  β, δ, Vg, Ko, Kg, Kl, cbarg = p

  nmax = mainparameters()

  du[1] = dn = 1 * cg / (cg + Kg) * 1 * β * n * (1 - n / nmax) - δ * n
  du[2] = dcg = -Vg * n * cg / (cg + cbarg)
  du[3] = dcl = -2 * (-Vg * n * cg / (cg + cbarg)) + 1 / 3 * (-Vo * n * co / (co + cbaro))
  du[4] = dco = 0
end

function neotissue3EQsConsOLG!(du, u, p, t)
  n, cg, cl, co = u
  β, δ, Vg, Ko, Kg, Kl, cbarg = p

  nmax = mainparameters()

  du[1] = dn = 1 * 1 * 1 * β * n * (1 - n / nmax) - δ * n
  du[2] = dcg = -Vg * n * cg / (cg + cbarg)
  du[3] = dcl = -2 * (-Vg * n * cg / (cg + cbarg)) + 1 / 3 * (-Vo * n * co / (co + cbaro))
  du[4] = dco = 0
end