"""
Based on
https://hyperopt.github.io/hyperopt/
"""
module TestLogQuantNormal

using Test
using TreeParzen

# Simple logquantnormal
# Minimize the objective over the space
space = Dict(:x => HP.LogQuantNormal(:x, 0.0, 1.0, 2.0))
best = fmin(params -> params[:x]^2, space, 100)
@test 0 <= best[:x] <= 1

# What if properties are the same?
space = Dict(:x => HP.LogQuantNormal(:x, 0.0, 1.0, 1.0))
fmin(params -> params[:x]^2, space, 100)
push("department")
push(space)


end # module TestLogQuantNormal
true
