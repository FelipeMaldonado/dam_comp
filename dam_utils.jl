#  Copyright 2018, Mehdi Madani
#  This Source Code Form is subject to the terms of the GNU GENERAL PUBLIC LICENSE version 3
#  If a copy of the GPL version 3 was not distributed with this
#  file, You can obtain one at http://www.gnu.org/licenses/gpl-3.0.en.html
#############################################################################


type damdata
  areas::DataFrame
  periods::DataFrame
  hourly::DataFrame
  mp_headers::DataFrame
  mp_hourly::DataFrame
  line_capacities::DataFrame
end

type damsol
  xval #hourly bids
  uval #on/off decisions for MP bids
  xhval #hourly bids associated to MP bids
  fval  #flows through lines
  sval  #surplus of hourly bids
  scval #surplus of MP bids
  shmaxval
  shminval
  vval
  priceval
  gupval
  gdownval
end

type damsolip
  xval #hourly bids
  uval #on/off decisions for MP bids
  xhval #hourly bids associated to MP bids
  fval  #flows through lines
  priceval
  stpriceval
end

function bidprofits(mydata, mysol)
  areas = Array(mydata.areas)
  periods = Array(mydata.periods)
  hourly = mydata.hourly
  mp_headers = mydata.mp_headers
  mp_hourly = mydata.mp_hourly
  line_capacities = mydata.line_capacities
  nbHourly = nrow(mydata.hourly)
  nbMp = nrow(mydata.mp_headers)
  nbMpHourly = nrow(mydata.mp_hourly)
  nbAreas = length(mydata.areas)
  nbPeriods = length(mydata.periods)
  rup = mydata.mp_headers[:,:RU]
  rdown = mydata.mp_headers[:,:RD]
  priceH = [mysol.priceval[mp_hourly[h, :LH], mp_hourly[h, :TH]] for h in 1:nbMpHourly]

  profits = zeros(Float64, nbMp)
  totalQ = zeros(Float64, nbMp)
  for c in 1:nbMp
    current_ = find( mp_hourly[:,:MP].== mp_headers[c,:MP])
    mp_hourly_current = mp_hourly[current_,:]
    priceH_current = priceH[current_]

    profit = dot(mysol.xhval[current_],(mp_hourly_current[:,:QH].data).*(mp_hourly_current[:,:PH].data - priceH_current)) - mp_headers[c,:FC]*mysol.uval[c]
    profits[c] = profit
    totalQ[c] = abs(dot(mysol.xhval[current_],(mp_hourly_current[:,:QH].data)))
  end
  return profits, totalQ
end

function bidprofits_max(mydata, mysol; ramp_activation = 0)
  areas = Array(mydata.areas)
  periods = Array(mydata.periods)
  hourly = mydata.hourly
  mp_headers = mydata.mp_headers
  mp_hourly = mydata.mp_hourly
  line_capacities = mydata.line_capacities
  nbHourly = nrow(mydata.hourly)
  nbMp = nrow(mydata.mp_headers)
  nbMpHourly = nrow(mydata.mp_hourly)
  nbAreas = length(mydata.areas)
  nbPeriods = length(mydata.periods)
  rup = mydata.mp_headers[:,:RU]
  rdown = mydata.mp_headers[:,:RD]
  priceH = [mysol.priceval[mp_hourly[h, :LH], mp_hourly[h, :TH]] for h in 1:nbMpHourly]

  profits = zeros(Float64, nbMp)
  totalQ = zeros(Float64, nbMp)
  for c in 1:nbMp
    m = Model(solver=CplexSolver(CPX_PARAM_EPGAP=1e-8, CPX_PARAM_EPAGAP=1e-8, CPX_PARAM_EPINT=1e-7, CPX_PARAM_TILIM=600, CPX_PARAM_BRDIR=-1, CPX_PARAM_HEURFREQ=-1))
    #println(length(find( mp_hourly[:,:MP].== mp_headers[c,:MP] )))
    current_ = find( mp_hourly[:,:MP].== mp_headers[c,:MP])
    mp_hourly_current = mp_hourly[current_,:]
    priceH_current = priceH[current_]

    nbxh = nrow(mp_hourly_current)
    @variable(m, u, Bin)
    @variable(m, xh[1:nbxh])
    @constraint(m, mp_control_upperbound[h=1:nbxh],
                    xh[h] <= u)
    @constraint(m, mp_control_lowerbound[h=1:nbxh],
                    xh[h] >= mp_hourly_current[h,:AR]*u)

    if ramp_activation == 1
      @constraint(m, rampup_constr[t in periods; !isna(rup[c]) && t != periods[end]],
        sum((-1)*xh[h]*mp_hourly_current[h, :QH]  for h in 1:nbxh if mp_hourly_current[h, :TH]==(t+1))
        - sum((-1)*xh[h]*mp_hourly_current[h, :QH]  for h in 1:nbxh if mp_hourly_current[h, :TH]==t)
        <= rup[c]*u)
      @constraint(m, rampdown_constr[t in periods; !isna(rdown[c]) && t != periods[end]],
        sum((-1)*xh[h]*mp_hourly_current[h, :QH]  for h in 1:nbxh if mp_hourly_current[h, :TH]==t)
        - sum((-1)*xh[h]*mp_hourly_current[h, :QH]  for h in 1:nbxh if mp_hourly_current[h, :TH]==(t+1))
        <= rdown[c]*u)
    end

    obj = dot(xh,(mp_hourly_current[:,:QH].data).*(mp_hourly_current[:,:PH].data - priceH_current)) - mp_headers[c,:FC]*u
    @objective(m, Max,  obj)
    status = solve(m)
    profit = getobjectivevalue(m)
    xhval_max = getvalue(xh)
    totalQ[c] = abs(dot(xhval_max,(mp_hourly_current[:,:QH].data))) #[current_]
    profits[c] = profit
  end
  return profits, totalQ
end

function countprb(mydata, mysol; ramp_activation = 0)
  tol = 1e-2
  maxprofits, totalQmax = bidprofits_max(mydata, mysol, ramp_activation = ramp_activation)
  profits, totalQ = bidprofits(mydata, mysol)
  #nbprb = count(i-> (i > tol), maxprofits-profits)
  z = Vector{Bool}(length(profits))
  for i in 1:length(profits)
    z[i] = maxprofits[i]-profits[i] > tol && totalQmax[i] - totalQ[i] > tol
    # z[i] = maxprofits[i]-profits[i] > tol && maxprofits[i] > tol
  end
  return sum(z)
end

function countpab(mydata, mysol) #; ramp_activation = 0
  tol = 1e-2
  profits, totalQ = bidprofits(mydata, mysol)
  nbpab = count(i-> (i < -tol), profits)
end
