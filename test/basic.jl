module TestBasic

using Test
using TreeParzen

# Quadratic TPE
objective(params) = (params[:x] - 3)^2
best = fmin(objective, Dict(:x => HP.Uniform(:x, -5.0, 5.0)), 50)
@test abs(best[:x] - 3.0) < .25

# Generate trials to calculate
objective(params) = params[:x]^2 + params[:y]^2
space = Dict(
    :x => HP.Uniform(:x, -10.0, 10.0),
    :y => HP.Uniform(:y, -10.0, 10.0),
)
points = [Dict(:x => 0.0, :y => 0.0), Dict(:x => 1.0, :y => 1.0)]
best = fmin(objective, space, 10, points)
@test best == Dict(:x => 0.0, :y => 0.0)

end
true
