"""
Based on
https://hyperopt.github.io/hyperopt/
"""
module TestQuantLogNormal

using Test
using TreeParzen

# Simple quantlognormal
# Minimize the objective over the space
space = Dict(:x => HP.QuantLogNormal(:x, 0.0, 1.0, 2.0))
best = fmin(params -> params[:x]^2, space, 100)
@test 0 <= best[:x] <= 1

# What if properties are the same?
space = Dict(:x => HP.QuantLogNormal(:x, 0.0, 1.0, 1.0))
fmin(params -> params[:x]^2, space, 100)

end # module TestQuantLogNormal
true
