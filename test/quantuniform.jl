"""
Based on
https://hyperopt.github.io/hyperopt/
"""
module TestQuantUniform

using Test
using TreeParzen

# Simple quantuniform
# Minimize the objective over the space
space = Dict(:x => HP.QuantUniform(:x, 0.0, 1.0, 2.0))
best = fmin(params -> params[:x]^2, space, 100)
@test 0 <= best[:x] <= 1

end # module TestQuantUniform
true
