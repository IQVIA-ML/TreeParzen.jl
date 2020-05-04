module TestPoints

using Test
using TreeParzen

objective(params) = params[:x]^2 + params[:y]^2
space = Dict(:x => HP.Uniform(:x, -10.0, 10.0), :y => HP.Uniform(:y, -10.0, 10.0))
points = [Dict(:x => 0.0, :y => 0.0), Dict(:x => 1.0, :y => 1.0)]

best = fmin(objective, space, 10, points)
@test iszero(best[:x])
@test iszero(best[:y])

# test to catch ArgumentError when number of iterations is lower than the number of points
@test_throws ArgumentError fmin(objective, space, 1, points)

end
true
