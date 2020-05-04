"""
Based on
https://hyperopt.github.io/hyperopt/
"""
module TestLogUniform

using StatsBase
using Test
using TreeParzen

# we do this because current version of StatsBase getting pulled in
# doesn't support edges keyword argument
# NOTE: IF YOU DONT HAVE -INF IN THE FIRST VALUE OF EDGES
# YOU MIGHT MISS SOME POINTS -- USE AT OWN RISK
# Each invocation is relatively memory intensive,
# don't use with millions of points (or use at own risk)
# When statsbase is available:
#   fit(StatsBase.Histogram, samples; edges=bin_edges).weights
function edgeshist(values, edges::Vector{})
       x = values .< edges'
       y = sum(x; dims=1)
       return diff(dropdims(y; dims=1))
end

logspace_edges(emin, emax, nbins) = exp.(emin : ((emax - emin) / nbins) : emax)


# test that distributions come out right
@testset "loguniform" begin

    N = 10_000
    COUNTMAX = 1200
    COUNTMIN = 800
    N_BINS = 10

    samples = [TreeParzen.Resolve.node(HP.LogUniform(:lu, -2.0, 2.0), TreeParzen.Trials.ValsDict()) for i in 1:N]
    bin_edges  = logspace_edges(-2.0, 2.0, N_BINS)

    @test length(samples) == N
    @test exp(-2) < minimum(samples)
    @test maximum(samples) < exp(2)
    h = edgeshist(samples, bin_edges)
    @test all(COUNTMIN .< h)
    @test all(h .< COUNTMAX)
end

end
true
