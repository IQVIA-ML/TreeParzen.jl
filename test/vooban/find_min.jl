"""
Based on:
https://medium.com/vooban-ai/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters-e3102814b919
"""
module TestFindMin

using Test
using TreeParzen

function objective_min(space)
    x = space[:x]
    y = space[:y]

    return x^2 + y^2
end

space = Dict(
    :x => HP.Uniform(:x, -5.0, 5.0),
    :y => HP.Uniform(:y, -5.0, 5.0)
)

best = fmin(objective_min, space, 100)
@test isapprox(best[:x], 0.013181950926553512, rtol = 0.2e1)
@test isapprox(best[:y], 0.0364742933684085, rtol = 0.2e1)

end
true
