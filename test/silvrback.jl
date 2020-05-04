"""
Based on
https://districtdatalabs.silvrback.com/parameter-tuning-with-hyperopt
"""
module TestSimple

using Test
using TreeParzen

# Simple 1
best = fmin(params -> params[:x], Dict(:x => HP.Uniform(:x, 0.0, 1.0)), 100)
@test isapprox(best[:x], 0.00076079610166447, rtol = 1e1)

# Simple 2
objective(params) = (params[:x] - 1)^2
best = fmin(objective, Dict(:x => HP.Uniform(:x, -2.0, 2.0)), 100)
@test isapprox(best[:x], 0.9990226896297454, rtol = 1e1)

# fspace
fspace = Dict(:x => HP.Uniform(:x, -5.0, 5.0))
objective(params) = params[:x]^2
best = fmin(objective, fspace, 50)
@test isapprox(best[:x], 0.035669063364728415, rtol = 1e1)

end
true
