module TestQuadratic

using Test
using TreeParzen

objective(params) = (params[:x] - 3)^2

best = fmin(objective, Dict(:x => HP.Uniform(:x, -5.0, 5.0)), 50)
@test abs(best[:x] - 3) < .25

end
true
