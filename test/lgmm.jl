module TestLGMM

using Statistics
using Test
import TreeParzen: GMM, LogGMM

# Log-normal mixture pdf at x > 0 (Hyperopt parameterisation: log(x) ~ Normal(mu, sigma^2)).
function lognormal_mixture_pdf(
    x::Real, weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64}
)::Float64
    s = 0.0
    for (w, mu, sigma) in zip(weights, mus, sigmas)
        z = (log(x) - mu) / sigma
        s += w * exp(-0.5 * z^2) / (x * sigma * sqrt(2pi))
    end
    return s
end

col(x) = reshape(collect(x), length(x), 1)

@testset "LGMM1" begin

    N_SAMPLES = 100_000

    # LGMM1 is exp(GMM1) in log-space; mean of log(draws) should match GMM1 on the same parameters.
    mixture = GMM.DistDetails([0.5, 0.5], [0.0, 1.0], [0.01, 0.01])
    log_draws = GMM.GMM1(mixture, N_SAMPLES)
    pos_draws = vec(LogGMM.LGMM1(mixture, N_SAMPLES))
    @test size(LogGMM.LGMM1(mixture, 10)) == (10, 1)
    @test all(pos_draws .> 0)
    @test isapprox(mean(log.(pos_draws)), mean(log_draws); rtol = 0.01)

    # Bounded draws stay in (exp(low), exp(high)) (half-open in log-space via GMM).
    low = 0.0
    high = 1.0
    bounded = vec(LogGMM.LGMM1(mixture, low, high, 5_000))
    @test all(bounded .>= exp(low))
    @test all(bounded .< exp(high))

    # lpdf: one log-normal component at x = 1
    one_component = GMM.DistDetails([1.0], [0.0], [1.0])
    llval = LogGMM.LGMM1_lpdf(col(1.0), one_component)
    @test size(llval) == (1,)
    @test isapprox(llval[1], log(1.0 / (1.0 * sqrt(2pi * 1.0^2))))

    # lpdf: mixture, two sample rows
    mixture = GMM.DistDetails([0.25, 0.25, 0.5], [0.0, 1.0, 2.0], [1.0, 2.0, 5.0])
    llval = LogGMM.LGMM1_lpdf(col([1.0, exp(0.5)]), mixture)
    @test size(llval) == (2,)
    @test isapprox(
        llval[1],
        log(lognormal_mixture_pdf(1.0, [0.25, 0.25, 0.5], [0.0, 1.0, 2.0], [1.0, 2.0, 5.0])),
    )
    @test isapprox(
        llval[2],
        log(lognormal_mixture_pdf(exp(0.5), [0.25, 0.25, 0.5], [0.0, 1.0, 2.0], [1.0, 2.0, 5.0])),
    )

end

end
true
