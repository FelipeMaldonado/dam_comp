#  Copyright 2018, Mehdi Madani
#  This Source Code Form is subject to the terms of the GNU GENERAL PUBLIC LICENSE version 3
#  If a copy of the GPL version 3 was not distributed with this
#  file, You can obtain one at http://www.gnu.org/licenses/gpl-3.0.en.html
#############################################################################

using JuMP,  DataFrames, DataArrays, CPLEX
include("dam_utils.jl")
tol = 1e-6

type ippricing_val
  sol::damsolip
  welfare::Float64
  nbNodes::Int64
  nbSolverCuts::Int64
  absgap::Float64
  runtime::Float64
  uplifts::Float64
  stprices
end


function  ippricing(mydata::damdata; ramp_activation = 0)

  pricecap_up = 3000 # market price range restrictions, upper bound,  in accordance to current European market rules (PCR)
  pricecap_down = -500 # market price range restrictions, lower bound, in accordance to current European market rules (PCR)


  solutions = Array{damsol}(1)
  solutions_u = Array{Vector{Float64}}(1)

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

  m = Model(solver=CplexSolver(CPX_PARAM_EPGAP=1e-8, CPX_PARAM_EPAGAP=1e-8, CPX_PARAM_EPINT=1e-7, CPX_PARAM_TILIM=600, CPX_PARAM_BRDIR=-1, CPX_PARAM_HEURFREQ=-1))


  @variable(m, 0<= x[1:nbHourly] <=1) # variables for steps of classical bid curves where no binary var is needed (hence index c is dropped) --> 'x_i'
  @variable(m, u[1:nbMp], Bin)        # variables u_c in the paper
  @variable(m, xh[1:nbMpHourly])      # variables x_ic in the paper. In the code here,  generally 'h' are used instead of 'i' (hc instead of ic in the paper, etc)
  @variable(m, 0<= f[areas, areas, periods] <= 0) # by default/at declaration, a line doesn't exist, though non-zero upper bounds on flows are set later for existing lines with non zero capacity

  for i in 1:nrow(line_capacities)
      setupperbound(f[line_capacities[i,:from], line_capacities[i,:too] , line_capacities[i,:t] ], line_capacities[i,:linecap] ) # setting transmission line capacities
  end

  @constraint(m, mp_control_upperbound[h=1:nbMpHourly, c=1:nbMp;  mp_hourly[h,:MP] == mp_headers[c,:MP]],
                  xh[h] <= u[c]) # constraint (8), dual variable is shmax[h]
  @constraint(m, mp_control_lowerbound[h=1:nbMpHourly, c=1:nbMp;  mp_hourly[h,:MP] == mp_headers[c,:MP]],
                  xh[h] >= mp_hourly[h,:AR]*u[c]) # constraint (9), dual variable is shmin[h]

  if ramp_activation == 1
    @constraint(m, rampup_constr[c in 1:nbMp, t in periods; !isna(rup[c]) && t != periods[end]],
      sum((-1)*xh[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==(t+1))
      - sum((-1)*xh[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==t)
      <= rup[c]*u[c] ) # constraints (12)
    @constraint(m, rampdown_constr[c in 1:nbMp, t in periods; !isna(rdown[c]) && t != periods[end]],
      sum((-1)*xh[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==t)
      - sum((-1)*xh[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==(t+1))
      <= rdown[c]*u[c] ) # constraints (13)
  end


  # Objective: maximizing welfare -> see (1) specialized to (6). N.B. Steps of classical bid curves (quantities QI and prices PI0) are treated separately
  obj = dot(x,(hourly[:,:QI].data).*(hourly[:,:PI0].data)) +  dot(xh,(mp_hourly[:,:QH].data).*(mp_hourly[:,:PH].data)) - dot(u, mp_headers[:, :FC])
  @objective(m, Max,  obj)

  # balance constraint (2). N.B. Here a basic 'transportation model' (so-called ATC model according to EU terminology) is used (and a simple two-nodes network in the datasets ...)
  @constraint(m, balance[loc in areas, t in periods],
  sum(x[i]*hourly[i, :QI] for i=1:nbHourly if hourly[i, :LI] == loc && hourly[i, :TI]==t ) # executed quantities of the 'hourly bids' for a given location 'loc' and time slot 't'
  +
  sum(xh[h]*mp_hourly[h, :QH] for h=1:nbMpHourly if mp_hourly[h,:LH] == loc && mp_hourly[h, :TH]==t ) # executed quantities of the 'hourly bids related to the MP bids' for a given location 'loc' and time slot 't'
  ==
  sum(f[loc_orig, loc, t] for loc_orig in areas if loc_orig != loc) - sum(f[loc, loc_dest,t] for loc_dest in areas if loc_dest != loc) # balance of inbound and outbound flows at a given location 'loc' at time 't'
  )


  status = solve(m) # solves the pure welfare maximizing program
  objval=getobjectivevalue(m)

  uval_=getvalue(u)
  xval_=getvalue(x)
  xhval_=getvalue(xh)
  fval_=getvalue(f)


  @constraint(m, ufix[c in 1:nbMp], u[c] == uval_[c]) # fixes the binary variables u_c to their (optimal) value

  # solves the "continuous relaxation" of the welfare max. prog. but which is indeed an LP as the u_c have fixed values
  # However, to extract dual var values below, one must here explicitely indicates "relaxation=true" when re-optimizing
  status = solve(m, relaxation = true)

#################################################################################################
  ufixdual = getdual(ufix) # get the startup prices as dual var to constraints (42)-(43)
  priceval_ = getdual(balance)
  stpricesval = ufixdual #.*uval_ # n.b. not needed as uplifts below are computed by mult. by uval_ but now keeps info on startup prices also for rejected bids

  mycandidate=damsolip(xval_, uval_, xhval_, fval_, priceval_, stpricesval)

  negstprices = (-1)*min(stpricesval,0)
  ipuplifts = sum(uval_.*negstprices)


ippricing_returned = ippricing_val(mycandidate, objval, 0, 0, 0, 0, ipuplifts, stpricesval)

end
