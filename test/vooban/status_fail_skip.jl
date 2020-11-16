"""
Based on
https://medium.com/vooban-ai/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters-e3102814b919
"""
module TestStatusFailSkip

using Test
using TreeParzen

function objective_fail(space)::Float64
    x = space[:x]
    y = space[:y]

    if y > 1
        # Make use of status fail as an example of skipping on error
        return Inf
    end
    return x^2 + y^2
end

space = Dict(
    :x => HP.Uniform(:x, -5.0, 5.0),
    :y => HP.Uniform(:y, -5.0, 5.0),
)
best = fmin(objective_fail, space, 100)
@test isapprox(best[:x], 0.051738577119312464, rtol = 4)
@test isapprox(best[:y], -0.09365785427863083, rtol = 4)

space = Dict(
    :x => HP.Uniform(:x, -5.0, 5.0),
)

# test to catch ArgumentError in evaluate hyperparams
@test_throws ArgumentError fmin(objective_fail, space, 10)

end
true
