module TestLinearForgettingWeights

using Compat
import Distributions
using Test

using TreeParzen

# just testing function itself

# N is a number of observations/mus, lf is based on config and indicates a positive linear forgetting value beyond which
# older points are de-weighted, e.g. if lf == 25, only values > 25 will be taken into consideration
# case 1: when N == 0 is it possible to achieve, or can there never be the case when N == 0? what can be its use case:
# no previous history?
case1 = TreeParzen.LinearForgettingWeights.linear_forgetting_weights(0, 1)
@test length(case1) == 0
# case 2: when N < lf
case2 = TreeParzen.LinearForgettingWeights.linear_forgetting_weights(6, 7)
@test minimum(case2) == 1.0
# case 3: when N - lf == 1
case3 = TreeParzen.LinearForgettingWeights.linear_forgetting_weights(8, 7)
@test minimum(case3) == 0.125
# case 4: when N - lf > 1
case4 = TreeParzen.LinearForgettingWeights.linear_forgetting_weights(4, 2)
@test minimum(case4) == 0.25
# case 5: when both N and lf == 0
case5 = TreeParzen.LinearForgettingWeights.linear_forgetting_weights(0, 0)
@test length(case5) == 0

end # module TestLinearForgettingWeights
true
