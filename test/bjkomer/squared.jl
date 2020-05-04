"""
Based on
https://github.com/bjkomer/hyperopt-tutorial/blob/master/intro.py
"""
module TestSquared

using Test
using TreeParzen

objective(params) = params[:x]^2
best = fmin(objective, Dict(:x => HP.Uniform(:x, -10.0, 10.0)), 100)
@test isapprox(best[:x], 0.005438783301373859, rtol=1e2)

end
true
