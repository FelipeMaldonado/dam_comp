#  Copyright 2018, Mehdi Madani
#  This Source Code Form is subject to the terms of the GNU GENERAL PUBLIC LICENSE version 3
#  If a copy of the GPL version 3 was not distributed with this
#  file, You can obtain one at http://www.gnu.org/licenses/gpl-3.0.en.html
#############################################################################

# pwd()  # to get your current workind dir

cd("/home/lxuser/devel/dam_comp/") # Set your working directory appropriately, for example: cd("/home/devel/dam_comp/")

using JuMP,  DataFrames, DataArrays, CPLEX
include("dam_utils.jl")
include("dam_europricing.jl")
include("dam_convexhullpricing.jl")
include("dam_ippricing.jl")

welfare_uplifts_comp = DataFrame(Inst = Int64[],Nb_MpBids=Float64[], Nb_Steps=Float64[], welfareCHP = Float64[], welfareIP = Float64[], welfareEU = Float64[], upliftsCHP = Float64[], upliftsIP = Float64[], runEU = Float64[], runIP = Float64[], runCHP = Float64[], pabEU = Int64[], prbEU = Int64[], pabIP = Int64[], prbIP = Int64[], pabCHP = Int64[], prbCHP = Int64[], accMicsEu = Int64[])

for ssid in 0:10

areas=readtable(string("./data/daminst-", ssid,"/areas.csv"))                   # list of bidding zones
periods=readtable(string("./data/daminst-", ssid,"/periods.csv"))               # list of periods considered
hourly=readtable(string("./data/daminst-", ssid,"/hourly_quad.csv"))            # classical demand and offer bid curves
mp_headers=readtable(string("./data/daminst-", ssid,"/mp_headers.csv"))         # MP bids: location, fixed cost, variable cost
mp_hourly=readtable(string("./data/daminst-", ssid,"/mp_hourly.csv"))           # the different bid curves associated to a MP bid
line_capacities=readtable(string("./data/daminst-", ssid,"/line_cap.csv"))      # tranmission capacities of lines. as of now, a simple "ATC" (i.e. transportation) model is used to describe the transmission netwwork, though any linear network model could be used

mydata = damdata(areas, periods, hourly, mp_headers, mp_hourly, line_capacities)
nbHourly = nrow(mydata.hourly)
nbMp = nrow(mydata.mp_headers)
nbMpHourly = nrow(mydata.mp_hourly)

runtime_euro = @elapsed eurosol = europricing(mydata, method_type = "benders_modern", ramp_activation = 1)
runtime_ip   = @elapsed ipsol = ippricing(mydata, ramp_activation = 1)
runtime_chp  = @elapsed chpsol = convexhullpricing(mydata, ramp_activation = 1) 

europrb = countprb(mydata, eurosol.sol, ramp_activation = 1)
europab = countpab(mydata, eurosol.sol)
ipprb = countprb(mydata, ipsol.sol, ramp_activation = 1)
ippab = countpab(mydata, ipsol.sol)
chprb = countprb(mydata, chpsol.sol, ramp_activation = 1)
chpab = countpab(mydata, chpsol.sol)

accmics = sum(eurosol.sol.uval)

push!(welfare_uplifts_comp, [ssid, nbMp,nbHourly + nbMpHourly, chpsol.welfare, ipsol.welfare, eurosol.welfare, chpsol.uplifts, ipsol.uplifts, runtime_euro, runtime_ip, runtime_chp, europab, europrb, ippab, ipprb, chpab, chprb, accmics])

welfare_uplifts_str = string("./welfares.csv")
writetable(welfare_uplifts_str, welfare_uplifts_comp)

end
