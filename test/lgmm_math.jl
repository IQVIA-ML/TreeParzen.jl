module TestLGMMMath

using Statistics
using Test
import TreeParzen: GMM, LogGMM

# Empirical vs LGMM1_lpdf on the positive scale (continuous, unbounded only).
# Quantised log paths are covered by sampler tests (logquantuniform, etc.)

weights = [0.1, 0.3, 0.4, 0.2]
mus = [1.0, 2.0, 3.0, 4.0]
sigmas = [0.1, 0.4, 0.8, 2.0]
mixture = GMM.DistDetails(weights, mus, sigmas)

col(x) = reshape(collect(x), length(x), 1)

log_samples = GMM.GMM1(mixture, 10_001)
pos_samples = sort(exp.(log_samples))
edges = pos_samples[1:500:end]
pdf = exp.(LogGMM.LGMM1_lpdf(col(edges[1:end - 1]), mixture))
dx = edges[2:end] .- edges[1:end - 1]
y = 1 ./ dx ./ length(dx)
err = (pdf .- y) .^ 2
@test maximum(err) < 0.1
@test mean(err) < 0.01
@test median(err) < 0.01

# Low and high (quantised path): empirical mass should match bounded LGMM1_lpdf.
low = 2.5
high = 3.5
q = 0.1
bounded_q_samples = vec(LogGMM.LGMM1(mixture, low, high, q, 10_001))
@test all(bounded_q_samples .>= exp(low))
@test all(bounded_q_samples .< exp(high))

vals = sort(unique(bounded_q_samples))
counts = [count(==(v), bounded_q_samples) for v in vals]
y = counts ./ length(bounded_q_samples)
prob = vec(exp.(LogGMM.LGMM1_lpdf(col(vals), mixture, low, high, q)))
err = (prob .- y) .^ 2
@test maximum(err) < 0.1
@test mean(err) < 0.01
@test median(err) < 0.01

@test_throws ArgumentError LogGMM.LGMM1(mixture, 3.5, 3.5, 10_001)

end
true
