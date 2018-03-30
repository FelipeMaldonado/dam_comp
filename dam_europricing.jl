#  Copyright 2018, Mehdi Madani
#  This Source Code Form is subject to the terms of the GNU GENERAL PUBLIC LICENSE version 3
#  If a copy of the GPL version 3 was not distributed with this
#  file, You can obtain one at http://www.gnu.org/licenses/gpl-3.0.en.html
#############################################################################

# Implementation of the models and algorithms described in
# 'Revisiting minimum profit conditions in uniform price day-ahead electricity auctions'
# European Journal of Operational Research, Volume 266, Issue 3, 1 May 2018, Pages 1072-1085

using JuMP,  DataFrames, DataArrays, CPLEX #, Cbc #, Gurobi
include("dam_utils.jl")


type europricing_val
  sol::damsol
  welfare::Float64
  nbNodes::Int64
  nbSolverCuts::Int64
  absgap::Float64
  runtime::Float64
end

# method_type = "benders_classic"    # 'classic' version of the Benders decomposition described in Section 5, see Theorems 5 and 6)
# method_type = "benders_modern"       # 'modern' version of the Benders decomposition described in Section 5, see Theorem 7. N.B. requires a solver supporting locally valid cuts callbacks
# method_type = "primal-dual"        # When "mic_activation" below is set to 0, it correponds to the formulation 'MarketClearing-MPC' (64)-(78) in the paper, given 'as is' to the solver
# method_type = "mp_relaxed"         # Solves the primal only (by decoupling it from the "dual"). Hence, MP conditions + other equilibrium conditions are not all enforced in that case. For comparison purposes.
# mic_activation = 0                   # if set to 1, it applies to the "primal-dual" formulation the modifications described in Section 3.3 to handle minimum income conditions as in OMIE-PCR, i.e. setting fixed costs to 0 in the objective, etc
# ramp_activation = 0


# N.B. Regarding the Benders decompositions:
# to check the global feasibility (is it in the projection 'G' ?) of a primal feasible solution as described in Theorem 3 (before adding the relevant cuts in case the candidate must be rejected),
# constraints (65) and (74)-(78) are checked by minimizing the right-hand side of (65) (the 'dual objective'), subject to constraints (74)-(78) where the values of the u_c given by the 'master program' are used (fixed).
# For fixed values of the u_c given by the 'master primal program', the terms 'bigM[c]*(1-u[c])' in (76) are included by modifying the right-hand constant terms of the constraints first declared withtout them. These right-hand terms are modified each time a new candidate provided by the master is considered.

function  europricing(mydata::damdata; method_type = "primal-dual", mic_activation = 0, ramp_activation = 0)

  if method_type != "primal-dual" &&  mic_activation != 0
    error("ERROR: OMIE complex orders with a minimum income condition only handled with the primal-dual method type")
  end

  pricecap_up = 3000 # market price range restrictions, upper bound,  in accordance to current European market rules (PCR)
  pricecap_down = -500 # market price range restrictions, lower bound, in accordance to current European market rules (PCR)


  solutions = Array{damsol}(1)
  solutions_u = Array{Vector{Float64}}(1)


  if mic_activation == 1
    FC_copy = mp_headers[:,:FC]
    mp_headers[:,:FC] = 0
  end


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

  # big-M's computation
  bigM = zeros(Float64, nbMp)

  for c in 1:nbMp
    bigM[c] -= mp_headers[c, :FC]
  end

  for h in 1:nbMpHourly
    bigM[find( mp_headers[:,:MP].== mp_hourly[h,:MP] )] += (pricecap_up - mp_hourly[h,:PH])*abs(mp_hourly[h,:QH]) #
  end

  m = Model(solver=CplexSolver(CPX_PARAM_EPGAP=1e-8, CPX_PARAM_EPAGAP=1e-8, CPX_PARAM_EPINT=1e-7, CPX_PARAM_TILIM=600, CPX_PARAM_BRDIR=-1, CPX_PARAM_HEURFREQ=-1))
  #, CPX_PARAM_EACHCUTLIM=0, CPX_PARAM_FRACCUTS=-1, CPX_PARAM_EACHCUTLIM=0, CPX_PARAM_LPMETHOD=2 , CPX_PARAM_THREADS=12 #CPX_PARAM_MIPDISPLAY=1, CPX_PARAM_MIPINTERVAL=1
  # m = Model(solver = GurobiSolver(BranchDir=-1, MIPGap=1e-9, MIPGapAbs=1e-9, IntFeasTol=1e-9, TimeLimit=600)) #Method=1,
  # m = Model(solver = CbcSolver(integerTolerance=1e-9, ratioGap=1e-9, allowableGap=1e-9 ))

  @variable(m, 0<= x[1:nbHourly] <=1) # variables denoted 'x_i' in the paper
  @variable(m, u[1:nbMp], Bin)        # variables u_c
  @variable(m, xh[1:nbMpHourly])      # variables x_hc, indexed below with the symbol 'h'
  @variable(m, 0<= f[areas, areas, periods] <= 0) # by default/at declaration, a line doesn't exist, though non-zero upper bounds on flows are set later for existing lines with non zero capacity

  for i in 1:nrow(line_capacities)
      setupperbound(f[line_capacities[i,:from], line_capacities[i,:too] , line_capacities[i,:t] ], line_capacities[i,:linecap] ) # setting transmission line capacities
  end

  @constraint(m, mp_control_upperbound[h=1:nbMpHourly, c=1:nbMp;  mp_hourly[h,:MP] == mp_headers[c,:MP]],
                  xh[h] <= u[c]) # constraint (3) or (67), dual variable is shmax[h]
  @constraint(m, mp_control_lowerbound[h=1:nbMpHourly, c=1:nbMp;  mp_hourly[h,:MP] == mp_headers[c,:MP]],
                  xh[h] >= mp_hourly[h,:AR]*u[c]) # constraint (4) or (68), dual variable is shmin[h]

  if ramp_activation == 1
    @constraint(m, rampup_constr[c in 1:nbMp, t in periods; !isna(rup[c]) && t != periods[end]],
      sum((-1)*xh[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==(t+1))
      - sum((-1)*xh[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==t)
      <= rup[c]*u[c] )
    @constraint(m, rampdown_constr[c in 1:nbMp, t in periods; !isna(rdown[c]) && t != periods[end]],
      sum((-1)*xh[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==t)
      - sum((-1)*xh[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==(t+1))
      <= rdown[c]*u[c] )
  end


  # Objective: maximizing welfare -> see (1) or (64)
  obj = dot(x,(hourly[:,:QI].data).*(hourly[:,:PI0].data)) +  dot(xh,(mp_hourly[:,:QH].data).*(mp_hourly[:,:PH].data)) - dot(u, mp_headers[:, :FC])
  @objective(m, Max,  obj)

  # balance constraint (6) or (70) specialized to a so-called 'ATC network model' (i.e. transportation model)
  @constraint(m, balance[loc in areas, t in periods],
  sum(x[i]*hourly[i, :QI] for i=1:nbHourly if hourly[i, :LI] == loc && hourly[i, :TI]==t ) # executed quantities of the 'hourly bids' for a given location 'loc' and time slot 't'
  +
  sum(xh[h]*mp_hourly[h, :QH] for h=1:nbMpHourly if mp_hourly[h,:LH] == loc && mp_hourly[h, :TH]==t ) # executed quantities of the 'hourly bids related to the MP bids' for a given location 'loc' and time slot 't'
  ==
  sum(f[loc_orig, loc, t] for loc_orig in areas if loc_orig != loc) - sum(f[loc, loc_dest,t] for loc_dest in areas if loc_dest != loc) # balance of inbound and outbound flows at a given location 'loc' at time 't'
  )


  if method_type != "primal-dual"
    mdual = Model(solver=CplexSolver(CPX_PARAM_LPMETHOD=2))
  # mdual = Model(solver = GurobiSolver(BranchDir=-1, MIPGap=1e-9, MIPGapAbs=1e-9, IntFeasTol=1e-9, TimeLimit=600)) #Method=1,
  # mdual = Model(solver = CbcSolver(integerTolerance=1e-9, ratioGap=1e-9, allowableGap=1e-9 ))
    else mdual = m
  end

  @variable(mdual, s[1:nbHourly] >=0) # variables s_i, corresponding to the economic surplus of the hourly order 'i', see Lemma 1
  @variable(mdual, sc[1:nbMp] >=0)    # variables s_c, corresponding to the overall economic surplus associated with the MP order 'c'
  @variable(mdual, shmax[1:nbMpHourly] >=0) # see Lemmas 2 and 3 for the interpretation of s^max_hc
  @variable(mdual, shmin[1:nbMpHourly] >=0) # see Lemmas 2 and 3 for the interpretation of s^min_hc
  @variable(mdual, price[areas, periods]) # price variables denoted pi_{l,t} in the paper
  @variable(mdual, v[areas, areas, periods] >=0) # dual variables of the line capacity constraints



  @constraint(mdual, hourlysurplus[i=1:nbHourly], s[i] + hourly[i,:QI]*price[hourly[i,:LI], hourly[i,:TI]] >= hourly[i,:QI]*hourly[i,:PI0]) # constraint (11) or (74)
  @constraint(mdual, flowdual[loc1 in areas, loc2 in areas, t in periods], price[loc2, t] - price[loc1, t] <= v[loc1, loc2, t]) # corresponds to constraints (14) or (77) once specialized to the transmission network considered here ('ATC')

  if ramp_activation == 0
    # why this block cannot be put after "obj >= 'dualobj expssion', in which case appropriate cc conditions not satisfied anymore. see if obj >= ... holds in that case, or what is wrong"
    @constraint(mdual, mphourlysurplus[h=1:nbMpHourly], (shmax[h] - shmin[h]) +  mp_hourly[h,:QH]*price[mp_hourly[h,:LH], mp_hourly[h,:TH]] == mp_hourly[h,:QH]*mp_hourly[h,:PH]) # constraint (12) or (75)
    if method_type != "primal-dual"
    # With the Benders decomposition approach (classic or modern), the term  bigM[c]*(1-u[c]) of constraint (76) is first 'omitted', but the right-hand constant term is adapted later accordingly, when solving this 'dual worker program', depending on the values of u_c, see N.B. lines 19-22 above regarding the Benders decompositions
      @constraint(mdual, mpsurplus[c=1:nbMp], sc[c] - sum((shmax[h] - mp_hourly[h,:AR]*shmin[h]) for h=1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c, :MP]) + mp_headers[c,:FC] >= 0)
    elseif method_type == "primal-dual"
    # constraint (76) if the 'MarketClearing-MPC' formulation is used 'as is':
    @constraint(mdual, mpsurplus[c=1:nbMp], sc[c] - sum((shmax[h] - mp_hourly[h,:AR]*shmin[h]) for h=1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c, :MP]) + mp_headers[c,:FC] + bigM[c]*(1-u[c]) >= 0)
    end

    periodsext = [0;periods]
    @variable(mdual, gup[1:nbMp, periodsext] >=0)
    @variable(mdual, gdown[1:nbMp, periodsext] >=0)
    @constraint(mdual, gupfix[c in 1:nbMp, t in periodsext], gup[c,t] == 0)
    @constraint(mdual, gdownfix[c in 1:nbMp, t in periodsext], gdown[c,t] == 0)
  end

  if method_type != "primal-dual"
    # With decompositions, as mentionned above (see lines 19-22), the right-hand side of (65) is minimized subject to (74)-(78) and then checked against the left-hand side, hence the present dual objective declaration
    #*line backup* @objective(mdual, Min, sum{s[i], i=1:nbHourly} +  sum(sc[c] for c=1:nbMp) + sum{v[loc1, loc2, t]*getupperbound(f[loc1, loc2, t]), loc1 in areas, loc2 in areas, t in periods})
    # warning JuMP update, message copied: Replace sum{v[loc1,loc2,t] * getupperbound(f[loc1,loc2,t]),loc1 in areas,loc2 in areas,t in periods} with sum((v[loc1,loc2,t] * getupperbound(f[loc1,loc2,t]) for t in periods) for loc1 in areas for loc2 in areas).
    @objective(mdual, Min, sum(s[i] for i=1:nbHourly) +  sum(sc[c] for c=1:nbMp) + sum(v[loc1, loc2, t]*getupperbound(f[loc1, loc2, t]) for loc1 in areas for loc2 in areas for t in periods))
  elseif method_type == "primal-dual"
    @constraint(m, obj >= sum(s[i] for i=1:nbHourly) +  sum(sc[c] for c=1:nbMp) + sum(v[loc1, loc2, t]*getupperbound(f[loc1, loc2, t]) for loc1 in areas for loc2 in areas for t in periods)) # constraint (65) if 'MarketClearing-MPC' is coded 'as is'
    if mic_activation == 1
      # in case the OMIE-PCR appraoch is considered, as detailed in Section 3.3, the 'ad-hoc' constraints (81) must be added:
      @constraint(mdual, MIC[c=1:nbMp], sc[c] - sum((mp_hourly[h,:QH]*mp_hourly[h,:PH]*xh[h]) for h=1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c, :MP]) - FC_copy[c] + sum((mp_hourly[h,:QH]*xh[h]*mp_headers[c,:VC]) for h=1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c, :MP]) + FC_copy[c]*(1-u[c]) >= 0) #
    end
  end




  if ramp_activation == 1
    periodsext = [0;periods]
    @variable(mdual, gup[1:nbMp, periodsext] >=0)
    @variable(mdual, gdown[1:nbMp, periodsext] >=0)
    @constraint(mdual, gupfix0[c in 1:nbMp], gup[c,0] == 0)
    @constraint(mdual, gdownfix0[c in 1:nbMp], gdown[c,0] == 0)
    @constraint(mdual, gupfixT[c in 1:nbMp], gup[c,periodsext[end]] == 0)
    @constraint(mdual, gdownfixT[c in 1:nbMp], gdown[c,periodsext[end]] == 0)

    for c in 1:nbMp
      if isna(rup[c])
        @constraint(mdual, gupna[p in periodsext], gup[c,p] == 0)
      end
      if isna(rdown[c])
        @constraint(mdual, gdownna[p in periodsext], gdown[c,p] == 0)
      end
    end

    goodc = [find( mp_headers[:,:MP].== mp_hourly[h,:MP]) for h in 1:nbMpHourly]
    @constraint(mdual, mphourlysurplus1[h=1:nbMpHourly], (shmax[h] - shmin[h])
                                                         + (mp_hourly[h,:QH]*gdown[goodc[h][1], mp_hourly[h,:TH]-1] - mp_hourly[h,:QH]*gup[goodc[h][1], mp_hourly[h,:TH]-1])
                                                         + (mp_hourly[h,:QH]*gup[goodc[h][1], mp_hourly[h,:TH]] - mp_hourly[h,:QH]*gdown[goodc[h][1], mp_hourly[h,:TH]])
                                                         +  mp_hourly[h,:QH]*price[mp_hourly[h,:LH], mp_hourly[h,:TH]] == mp_hourly[h,:QH]*mp_hourly[h,:PH]) # constraint (12) or (75)

    if method_type != "primal-dual"
    @constraint(mdual, mpsurplus[c=1:nbMp], sc[c]
                                          - sum((shmax[h] - mp_hourly[h,:AR]*shmin[h]) for h=1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c, :MP])
                                          - sum(rup[c]*gup[c,t] for t in periodsext if !isna(rup[c]))
                                          - sum(rdown[c]*gdown[c,t] for t in periodsext if !isna(rdown[c]))
                                          + mp_headers[c,:FC] >= 0)
    elseif method_type == "primal-dual"
    @constraint(mdual, mpsurplus[c=1:nbMp], sc[c]
                                          - sum((shmax[h] - mp_hourly[h,:AR]*shmin[h]) for h=1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c, :MP])
                                          - sum(rup[c]*gup[c,t] for t in periodsext if !isna(rup[c]))
                                          - sum(rdown[c]*gdown[c,t] for t in periodsext if !isna(rdown[c]))
                                          + mp_headers[c,:FC] + bigM[c]*(1-u[c]) >= 0)
    end

  end


  cut_count = [0]
  tol=1e-5

  if method_type == "benders_modern"
    function workercb(cb)
      uval=getvalue(u)
      for c in 1:nbMp
        JuMP.setRHS(mpsurplus[c], - bigM[c]*(1-uval[c]) - mp_headers[c,:FC] ) # used to write constraints (76) with the values u_c given by the master program, by modifying the right-hand side constant term, see comments at lines 19-22
      end
      statusdual = solve(mdual)
      dualobjval = getobjectivevalue(mdual)
      objval=cbgetnodeobjval(cb)
      if dualobjval > objval + tol
        println("This primal solution is not valid, and is cut off ...")
        @lazyconstraint(cb, sum((1-u[c]) for c=1:nbMp if uval[c] == 1) + sum(u[c] for c=1:nbMp if uval[c] == 0)  >= 1)   # globally valid 'no-good cut', see Theorem 5
        @lazyconstraint(cb, sum((1-u[c]) for c=1:nbMp if uval[c] == 1)  >= 1, localcut=true)   # locally valid strengthened cut, see Theorem 7
        cut_count[1] += 1
        println("Cuts added: ", cut_count[1])
        println("node:", getnodecount(m))
      elseif dualobjval <= objval + tol
        fill!(solutions_u, uval)
      end
    end
  addlazycallback(m, workercb)
  end

  status = solve(m)
  objval=getobjectivevalue(m)

  # Once optimization is over, prices are re-computed ex-post (they could also be saved from within the callback, but let us note that all we need to recoonstruct them are the values of the u_c of a given solution)
  uval_c = getvalue(u)
  if method_type == "benders_modern"
    for c in 1:nbMp
      JuMP.setRHS(mpsurplus[c], - bigM[c]*(1-uval_c[c]) - mp_headers[c,:FC] ) # used to write constraints (76) with the values u_c given by the master program, by modifying the right-hand side constant term, see comments at lines 19-22
    end
    statusdual = solve(mdual)
  end

  if method_type == "benders_classic" # As of now, no time limit is specified here for the classic version ...
      uval=getvalue(u)
      tol=1e-4
      for c=1:nbMp
        JuMP.setRHS(mpsurplus[c], - bigM[c]*(1-uval[c]) - mp_headers[c,:FC] )
      end
      statusdual = solve(mdual)
      dualobjval = getobjectivevalue(mdual)
      while objval + tol < dualobjval
        @constraint(m, sum((1-u[c]) for c=1:nbMp if uval[c] == 1)  >= 1)
        cut_count[1] +=1
        status = solve(m)
        objval=getobjectivevalue(m)
        uval=getvalue(u)
        for c=1:nbMp
          JuMP.setRHS(mpsurplus[c], - bigM[c]*(1-uval[c]) - mp_headers[c,:FC] ) # used to write constraints (76) with the values u_c given by the master program, by modifying the right-hand side constant term, see comments at lines 19-22
        end
        statusdual = solve(mdual)
        dualobjval = getobjectivevalue(mdual)

        println("Primal Obj:", objval)
        println("Dual Obj:", dualobjval)
      end
  end

  if method_type == "mp_relaxed" # Re-compute prices for a primal solution in the case MP conditions have been relaxed. No guarentee is given then that all MP conditions *and* other equilibrium conditions are all enforced. For comparison purposes.
    uval=getvalue(u)
    tol=1e-4
    for c=1:nbMp
      JuMP.setRHS(mpsurplus[c], - bigM[c]*(1-uval[c]) - mp_headers[c,:FC] )
    end
    statusdual = solve(mdual)
    dualobjval = getobjectivevalue(mdual)
  end

  uval_=getvalue(u)
  xval_=getvalue(x)
  xhval_=getvalue(xh)
  fval_=getvalue(f)
  sval_=getvalue(s)
  scval_=getvalue(sc)
  shmaxval_=getvalue(shmax)
  shminval_=getvalue(shmin)
  priceval_=getvalue(price)
  vval_=getvalue(v)
  gupval_ =getvalue(gup)
  gdownval_ =getvalue(gdown)
  mycandidate=damsol(xval_, uval_, xhval_, fval_, sval_, scval_, shmaxval_, shminval_, vval_, priceval_, gupval_, gdownval_)

  nbNodes = MathProgBase.getnodecount(m)
  # Counting the number of cuts of all kinds generated by Cplex, safe user cuts which are internatlly numbered '15' :
  nbSolverCuts = mapreduce(i->CPLEX.get_num_cuts(m.internalModel.inner, i), +, [1:14;]) + mapreduce(i->CPLEX.get_num_cuts(m.internalModel.inner, i), +, [16:18;]) # sole cplex dependent feature
  absgap=MathProgBase.getobjbound(m) - getobjectivevalue(m)
  runtime = MathProgBase.getsolvetime(m)

europricing_returned = europricing_val(mycandidate, objval, nbNodes, nbSolverCuts, absgap, runtime)

end


function incomeCheck(mydata::damdata, mysol::damsol)
  areas = mydata.areas
  periods = mydata.periods
  hourly = mydata.hourly
  mp_headers = mydata.mp_headers
  mp_hourly = mydata.mp_hourly
  line_capacities = mydata.line_capacities

  areas=Array(areas)
  periods=Array(periods)
  periodsext = [0;periods]

  nbHourly=nrow(hourly)
  nbMp=nrow(mp_headers)
  nbMpHourly=nrow(mp_hourly)
  nbAreas=length(areas)
  nbPeriods=length(periods)

  xval = convert(Vector{Float64}, mysol.xval)
  uval = convert(Vector{Float64}, mysol.uval)
  xhval = convert(Vector{Float64}, mysol.xhval)
  sval = convert(Vector{Float64}, mysol.sval)
  scval = convert(Vector{Float64}, mysol.scval)
  shmaxval = convert(Vector{Float64}, mysol.shmaxval)
  shminval = convert(Vector{Float64}, mysol.shminval)
  #priceval = convert( Vector{Float64} , mysol.priceval)
  priceI = [mysol.priceval[hourly[i, :LI], hourly[i, :TI]] for i in 1:nbHourly]
  priceI = convert(Vector{Float64}, priceI)
  priceH = [mysol.priceval[mp_hourly[h, :LH], mp_hourly[h, :TH]] for h in 1:nbMpHourly]
  priceH = convert(Vector{Float64}, priceH)
  gupval = mysol.gupval
  gdownval = mysol.gdownval

  income_real = zeros(nbMp)
  income_linear = scval + mp_headers[:,:FC].*uval
  fixedcosts = mp_headers[:,:FC].*uval
  marginalcosts = zeros(nbMp)
  sc_values = [scval[c] for c in 1:nbMp]
  shmax_values = zeros(nbMp)
  shmin_AR_values = zeros(nbMp)
for c in 1:nbMp
  for h in 1:nbMpHourly
    if(mp_hourly[h,:MP] == mp_headers[c, :MP])
      income_real[c] -= xhval[h]*mp_hourly[h,:QH]*priceH[h]
      income_linear[c] -= xhval[h]*mp_hourly[h,:QH]*mp_hourly[h,:PH]
      marginalcosts[c] -= xhval[h]*mp_hourly[h,:QH]*mp_hourly[h,:PH]
      shmax_values[c]  += shmaxval[h]*uval[c]
      shmin_AR_values[c]  += shminval[h]*mp_hourly[h,:AR]*uval[c]
    end
  end
end
  #rampingsurplus = [mp_headers[c,:RU]*sum([gupval[c,t] for t in periodsext]) + mp_headers[c,:RD]*sum([gdownval[c,t] for t in periodsext])  for c in 1:nbMp]
  rampingsurplusup = [mp_headers[c,:RU]*sum([gupval[c,t] for t in periodsext]) for c in 1:nbMp]
  rampingsurplusdown = [mp_headers[c,:RD]*sum([gdownval[c,t] for t in periodsext])  for c in 1:nbMp]
  rampingsurplus = rampingsurplusup + rampingsurplusdown

  PL = income_real - fixedcosts - marginalcosts
  incomes = DataFrame([income_real income_linear fixedcosts marginalcosts sc_values PL shmax_values shmin_AR_values rampingsurplus rampingsurplusup rampingsurplusdown]);
  #incomes = DataFrame([income_real income_linear fixedcosts marginalcosts sc_values PL shmax_values shmin_AR_values]);
  rename!(incomes, :x1, :income_real)
  rename!(incomes, :x2, :income_linear)
  rename!(incomes, :x3, :fixedcosts)
  rename!(incomes, :x4, :marginalcosts)
  rename!(incomes, :x5, :sc_values)
  rename!(incomes, :x6, :ProfitsLosses)
  rename!(incomes, :x7, :sum_shmax)
  rename!(incomes, :x8, :sum_shmin_AR)
  rename!(incomes, :x9, :RampingSurplus)
  rename!(incomes, :x10, :RampingSurplusUP)
  rename!(incomes, :x11, :RampingSurplusDOWN)
  return incomes
end

function solQstats(mydata::damdata, mysol::damsol)
  areas = mydata.areas
  periods = mydata.periods
  hourly = mydata.hourly
  mp_headers = mydata.mp_headers
  mp_hourly = mydata.mp_hourly
  line_capacities = mydata.line_capacities

  areas=Array(areas)
  periods=Array(periods)
  periodsext = [0;periods]

  nbHourly=nrow(hourly)
  nbMp=nrow(mp_headers)
  nbMpHourly=nrow(mp_hourly)
  nbAreas=length(areas)
  nbPeriods=length(periods)


  xval = convert(Vector{Float64}, mysol.xval)
  uval = convert(Vector{Float64}, mysol.uval)
  xhval = convert(Vector{Float64}, mysol.xhval)
  sval = convert(Vector{Float64}, mysol.sval)
  scval = convert(Vector{Float64}, mysol.scval)
  shmaxval = convert(Vector{Float64}, mysol.shmaxval)
  shminval = convert(Vector{Float64}, mysol.shminval)
  #priceval = convert( Vector{Float64} , mysol.priceval)
  priceI = [mysol.priceval[hourly[i, :LI], hourly[i, :TI]] for i in 1:nbHourly]
  priceI = convert(Vector{Float64}, priceI)
  priceH = [mysol.priceval[mp_hourly[h, :LH], mp_hourly[h, :TH]] for h in 1:nbMpHourly]
  priceH = convert(Vector{Float64}, priceH)
  gupval = mysol.gupval
  gdownval = mysol.gdownval

fval_loc = Array{Float64}(nrow(line_capacities))
vval_loc = Array{Float64}(nrow(line_capacities))
Networkslack_d = Array{Float64}(nrow(line_capacities))
for i in 1:nrow(line_capacities)
      fval_loc[i] = mysol.fval[line_capacities[i,:from], line_capacities[i,:too] , line_capacities[i,:t] ]
      vval_loc[i] = mysol.vval[line_capacities[i,:from], line_capacities[i,:too] , line_capacities[i,:t] ]
      Networkslack_d[i] = vval_loc[i] - mysol.priceval[line_capacities[i,:too], line_capacities[i,:t]] +  mysol.priceval[line_capacities[i,:from], line_capacities[i,:t]]
end


  ### PRIMAL SLACKS TESTING
  Islack_p = (1-xval)
  maxminslack_hourly_p=maximum(min(sval, Islack_p))

  uval_match_xh = [ uval[find( mp_headers[:,:MP].== mp_hourly[h,:MP] ) ][1]  for h in 1:nbMpHourly]

  Hslack_max_p = (uval_match_xh - xhval)
  Hslack_max_p = convert(Vector{Float64}, Hslack_max_p)
  maxminslack_xh_max = maximum(min(shmaxval, Hslack_max_p))

  Hslack_min_p = (xhval - uval_match_xh.*mp_hourly[:,:AR])
  Hslack_min_p = convert(Vector{Float64}, Hslack_min_p)
  maxminslack_xh_min = maximum(min(shminval, Hslack_min_p))


  Cslack_p = (1-uval)
  maxminslack_mp_p=maximum(min(scval, Cslack_p))

  Networkslack_p = (line_capacities[:,:linecap] - fval_loc)
  maxminslack_network_p = maximum(min(vval_loc, Networkslack_p))


  mp_headers[:,:RU] = convert(Vector{Float64}, mp_headers[:,:RU])
  rampupslack_comp = mp_headers[:,:RU].*uval
  rampupslack = Array(Float64, (nbMp,nbPeriods))
  rampupslack_max = 0
  for c in 1:nbMp
    for t in 1:nbPeriods
      rampupslack[c,t] = rampupslack_comp[c]
      for h in 1:nbMpHourly
        #if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==(t+1)
        #  rampupslack[c,t] -= (-1)*xhval[h]*mp_hourly[h, :QH]
        #end
        if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==t
          rampupslack[c,t] += (-1)*xhval[h]*mp_hourly[h, :QH]
        end
      end
      rampupslack_max = max(rampupslack_max, abs(min(rampupslack[c,t], gupval[c,t])))
      #print(min(rampupslack[c,t], gupval[c,t]))
      #rampupslack[c,t] =
                        #rampupslack[c,t]
                        #- sum((-1)*xhval[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==(t+1))
                        #+ sum((-1)*xhval[h]*mp_hourly[h, :QH]  for h in 1:nbMpHourly if mp_hourly[h,:MP] == mp_headers[c,:MP] &&  mp_hourly[h, :TH]==t)
    end
  end

  print(rampupslack_max)
  #rampdownslack


  ### DUAL SLACKS TESTING
  Islack_d = sval - hourly[:,:QI].*(hourly[:,:PI0] - priceI)
  maxminslack_hourly_d=maximum(min(xval, Islack_d))

  surp_hmaxmin = zeros(nbMp)
  for h in 1:nbMpHourly
    for c in 1:nbMp
      if(mp_hourly[h,:MP] == mp_headers[c, :MP])
        surp_hmaxmin[c] += shmaxval[h] - mp_hourly[h,:AR]*shminval[h]
      end
    end
  end

  rampingsurplus = [mp_headers[c,:RU]*sum([gupval[c,t] for t in periodsext]) + mp_headers[c,:RD]*sum([gdownval[c,t] for t in periodsext])  for c in 1:nbMp]

  Cslack_d = scval  - surp_hmaxmin + mp_headers[:,:FC] - rampingsurplus
  Cslack_d = convert(Vector{Float64}, Cslack_d)
  maxminslack_mp_d=maximum(min(uval, Cslack_d))

  maxminslack_network_d = maximum(min(fval_loc, Networkslack_d))

############################ Quality summary
maxSlackViolation = max(maxminslack_hourly_p , maxminslack_hourly_d, maxminslack_xh_max, maxminslack_xh_min, maxminslack_mp_p, maxminslack_mp_d, maxminslack_network_p, maxminslack_network_d)
solstats = DataFrame([maxSlackViolation maxminslack_hourly_p  maxminslack_hourly_d maxminslack_xh_max maxminslack_xh_min maxminslack_mp_p maxminslack_mp_d maxminslack_network_p maxminslack_network_d])
rename!(solstats, :x1, :maxSlackViolation)
rename!(solstats, :x2, :maxminslack_hourly_p)
rename!(solstats, :x3, :maxminslack_hourly_d)
rename!(solstats, :x4, :maxminslack_xh_max)
rename!(solstats, :x5, :maxminslack_xh_min)
rename!(solstats, :x6, :maxminslack_mp_p)
rename!(solstats, :x7, :maxminslack_mp_d)
rename!(solstats, :x8, :maxminslack_network_p)
rename!(solstats, :x9, :maxminslack_network_d)
return solstats
end
