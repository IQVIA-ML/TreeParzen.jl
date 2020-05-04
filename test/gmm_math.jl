module TestGMMMath

using Compat
using Statistics
using Test
import TreeParzen: GMM

# GMM1 Math
weights = [0.1, 0.3, 0.4, 0.2]
mus = [1.0, 2.0, 3.0, 4.0]
sigmas = [0.1, 0.4, 0.8, 2.0]

# No low or high
samples = GMM.GMM1(weights, mus, sigmas, 10_001)
samples = sort(samples)
edges = samples[1:500:end]
pdf = exp.(GMM.GMM1_lpdf(edges[1:end - 1], weights, mus, sigmas))
dx = edges[2:end] .- edges[1:end - 1]
y = 1 ./ dx ./ length(dx)
err = (pdf .- y) .^ 2
@test maximum(err) < .1
@test mean(err) < .01
@test median(err) < .01

# Low and high
samples = GMM.GMM1(weights, mus, sigmas, 2.5, 3.5, 10_001)
samples = sort(samples)
edges = samples[1:500:end]
pdf = exp.(GMM.GMM1_lpdf(edges[1:end - 1], weights, mus, sigmas, 2.5, 3.5))
dx = edges[2:end] .- edges[1:end - 1]
y = 1 ./ dx ./ length(dx)
err = (pdf .- y) .^ 2
@test maximum(err) < 0.1
@test mean(err) < 0.01
@test median(err) < 0.01

# Low is >= high
@test_throws ArgumentError GMM.GMM1(weights, mus, sigmas, 3.5, 3.5, 10_001)

# QGMM1 Math
weights_t = [0.1, 0.3, 0.4, 0.2]
mus_t = [1.0, 2.0, 3.0, 4.0]
sigmas_t = [0.1, 0.4, 0.8, 2.0]
n_samples_t = 1_001

function test_samples(samples, c)
    @test samples == Int.(samples)
    samples_min = minimum(samples)
    samples_max = maximum(samples)
    bincount = samples .- samples_min
    counts = [count(x -> x == i, bincount) for i in 0:maximum(bincount)]
    @test sum(counts) == c.n_samples
    xcoords = @compat range(samples_min, samples_max; length = length(counts)) * c.q
    prob = if :low in propertynames(c)
        exp.(GMM.GMM1_lpdf(xcoords |> collect, c.weights, c.mus, c.sigmas, c.low, c.high, c.q))
    else
        exp.(GMM.GMM1_lpdf(xcoords |> collect, c.weights, c.mus, c.sigmas, c.q))
    end
    y = counts ./ c.n_samples
    err = (prob .- y) .^ 2

    @test maximum(err) < 0.1
    @test mean(err) < 0.01
    @test median(err) < 0.01
end

for c in (
    (weights = weights_t, mus = mus_t, sigmas = sigmas_t, q = 1.0, n_samples = n_samples_t),
    (weights = weights_t, mus = mus_t, sigmas = sigmas_t, q = 2.0, n_samples = n_samples_t),
    (weights = weights_t, mus = mus_t, sigmas = sigmas_t, q = 0.5, n_samples = n_samples_t),
)
    samples = GMM.GMM1(c.weights, c.mus, c.sigmas, c.q, c.n_samples) / c.q
    test_samples(samples, c)
end

for c in (
    (weights = weights_t, mus = mus_t, sigmas = sigmas_t, q = 1.0, low = 2.0, high = 4.0,
    n_samples = n_samples_t),
    (weights = weights_t, mus = mus_t, sigmas = sigmas_t, q = 2.0, low = 2.0, high = 4.0,
    n_samples = n_samples_t),
    (weights = weights_t, mus = mus_t, sigmas = sigmas_t, q = 1.0, low = 1.0, high = 4.1,
    n_samples = n_samples_t),
    (weights = weights_t, mus = mus_t, sigmas = sigmas_t, q = 2.0, low = 1.0, high = 4.1,
    n_samples = n_samples_t),
    (weights = [0.14285714, 0.28571429, 0.28571429, 0.28571429], mus = [5.505, 7., 2., 10.],
    sigmas = [8.99, 5., 8., 8.], q = 1.0, low = 1.01, high = 10.0, n_samples = 10_000),
    (weights = [0.33333333, 0.66666667], mus = [5.505, 5.], sigmas = [8.99, 5.19], q = 1.0,
    low = 1.01, high = 10.0, n_samples = 10_000),
)
    samples = GMM.GMM1(c.weights, c.mus, c.sigmas, c.low, c.high, c.q, c.n_samples) / c.q
    test_samples(samples, c)
end

end
true
