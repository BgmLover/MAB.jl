# This file include the utils for starting an experiment
# Use ```include("utils.jl")``` in your experiments

import Base: run
using Requires
@require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" using Plots
include("BanditExpBase.jl")
using ...Algorithms
using ...Arms
using ...Algorithms: make_agents_with_k
# using ..BanditExpBase: run