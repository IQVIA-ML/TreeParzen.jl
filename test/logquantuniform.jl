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


    # get their max and min so we don't really need to check against a set
    sample_vals = sort(unique(qlu_samples))

    samples = log.(sample_vals)
    gap = round.(diff(samples); digits=1)
    @test length(unique(gap)) == 1
    @test unique(gap)[1] == q
end


end # module TestLogQuantUniform
true
