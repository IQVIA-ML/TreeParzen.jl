module TestUniform

using StatsBase
using Test
using TreeParzen
import TreeParzen: API, Trials


N = 10_000
COUNTMAX = 1300
COUNTMIN = 700

samples = [TreeParzen.Resolve.node(HP.Uniform(:u, 0.0, 10.0), TreeParzen.Trials.ValsDict()) for i in 1:N]

@test minimum(samples) > 0
@test maximum(samples) < 10
h = fit(StatsBase.Histogram, samples, nbins=10).weights
@test all(COUNTMIN .< h)
@test all(h .< COUNTMAX)

end
true
