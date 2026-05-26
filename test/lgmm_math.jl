module TestLGMMMath

using Statistics
using Test
import TreeParzen: GMM, LogGMM

# Empirical vs LGMM1_lpdf on the positive scale (continuous, unbounded only).
# Quantised log paths are covered by sampler tests (logquantuniform, etc.)

weights = [0.1, 0.3, 0.4, 0.2]
mus = [1.0, 2.0, 3.0, 4.0]
sigmas = [0.1, 0.4, 0.8, 2.0]

const col = x -> reshape(collect(x), length(x), 1)

log_samples = GMM.GMM1(weights, mus, sigmas, 10_001)
pos_samples = sort(exp.(log_samples))
edges = pos_samples[1:500:end]
pdf = exp.(LogGMM.LGMM1_lpdf(col(edges[1:end - 1]), weights, mus, sigmas))
dx = edges[2:end] .- edges[1:end - 1]
y = 1 ./ dx ./ length(dx)
err = (pdf .- y) .^ 2
@test maximum(err) < 0.1
@test mean(err) < 0.01
@test median(err) < 0.01

@test_throws ArgumentError LogGMM.LGMM1(weights, mus, sigmas, 3.5, 3.5, 10_001)

end
true
