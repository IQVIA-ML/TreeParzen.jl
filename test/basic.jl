module TestBasic

using Test
using TreeParzen

# Generate trials to calculate
objective(params) = params[:x]^2 + params[:y]^2
space = Dict(:x => HP.Uniform(:x, -10.0, 10.0), :y => HP.Uniform(:y, -10.0, 10.0))
points = [Dict(:x => 0.0, :y => 0.0), Dict(:x => 1.0, :y => 1.0)]
best = fmin(objective, space, 10, points)
@test best == Dict(:x => 0.0, :y => 0.0)

end
true
