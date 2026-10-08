"""
Based on
https://hyperopt.github.io/hyperopt/
"""
module TestLogQuantUniform

using Test
using TreeParzen

@testset "quantised log uniform" begin
    q = 0.2
    qlu = HP.LogQuantUniform(:qlu, log(1.0), log(100.0), q)

    N = 10_000

    qlu_samples = [TreeParzen.Resolve.node(qlu, TreeParzen.Trials.ValsDict()) for i in 1:N]
    @test 1.0 <= minimum(qlu_samples)
    @test maximum(qlu_samples) <= 100.0

    sample_vals = sort(unique(qlu_samples))
    samples = log.(sample_vals)
    gap = round.(diff(samples); digits=1)

    @test length(unique(gap)) == 1
    @test unique(gap)[1] == q
end

@testset "adaptive logquantuniform posterior sampler" begin
    obs = TreeParzen.ConstrainedVectors.Observations([0.1, 1.0, 5.0])
    draws, mixture = TreeParzen.Samplers.logquantuniform(
        obs, -2.0, 2.0, 0.2, 500, Config()
    )

    @test length(draws) == 500
    # Quantization follows exponentiation, so values may extend by q/2 beyond the bounds.
    @test all(
        (draws.v .>= max(0.0, exp(-2.0) - 0.1)) .& (draws.v .<= exp(2.0) + 0.1)
    )
    @test all(isapprox.(draws.v ./ 0.2, round.(draws.v ./ 0.2); atol=1e-10))
    @test sum(mixture.weights) ≈ 1.0
end

end # module TestLogQuantUniform
true
