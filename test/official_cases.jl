"""
Based on
https://hyperopt.github.io/hyperopt/
"""
module TestOfficialCases

using Test
using TreeParzen

# Simple lognormal
# Minimize the objective over the space
best = fmin(params -> params[:x]^2, Dict(:x => HP.LogNormal(:x, 0.0, 1.0)), 100)
@test 0 <= best[:x] <= 1

# Basic case

# Define an objective function
function objective(params)
    case, val = params[:args]
    if case == :case1
        return val
    end

    return val^2
end

# Define a search space
space = Dict(
    :args => HP.Choice(:a, [
        (:case1, 1 + HP.LogNormal(:c1, 0.0, 1.0)),
        (:case2, HP.Uniform(:c2, -10.0, 10.0))
    ])
)

# Minimize the objective over the space
best = fmin(objective, space, 100)

if first(best[:args]) == :case1
    @test isapprox(last(best[:args]), 8.146095517525719e-17, rtol = 1e1)
elseif first(best[:args]) == :case2
    @test isapprox(last(best[:args]), 0.051871771726651195, rtol = 1e1)
else
    @test false
end

# Simplest case
space = Dict(:x => HP.Uniform(:x, -10.0, 10.0))
best = fmin(params -> params[:x]^2, space, 100)
@test isapprox(best[:x], 0.017044885261127796, rtol = 1e1)

end
true
