"""
Based on:
https://medium.com/vooban-ai/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters-e3102814b919
"""
module TestVoobanBasic

using Test
using TreeParzen

objective(params) = params[:x]^2 - params[:x] + 1
space = Dict(:x => HP.Uniform(:x, -5.0, 5.0))
best = fmin(objective, space, 100)
@test isapprox(best[:x], 0.5366692707330428, rtol=1e1)

end
true
