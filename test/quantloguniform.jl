"""
Based on
https://hyperopt.github.io/hyperopt/
"""
module TestQuantLogUniform

using Test
using TreeParzen

@testset "quantised log uniform" begin

    qlu = HP.QuantLogUniform(:qlu, log(1.01), log(20.0), 2.0)
    N = 10_000

    qlu_samples = [TreeParzen.Resolve.node(qlu, TreeParzen.Trials.ValsDict()) for i in 1:N]

    @test all(qlu_samples .% 2 .== 0)
    @test minimum(qlu_samples) == 2
    @test maximum(qlu_samples) == 20

    # get the distinct samples, we already asserted they are all mod 2
    # and their max and min so we don't really need to check against a set
    sample_vals = sort(unique(qlu_samples))'
    counts = sum(sample_vals .== qlu_samples; dims=1)
    # reverse sort because we expect biggest first and smallest last
    ordered_indices = sortperm(dropdims(counts; dims=1); rev=true)

    # check that its always ordered from bigger to smaller;
    # it should be fairly rare for one to get bigger than one preceeding it
    # but we allow it by adding [[1, -1, 2]] to the diffcheck, which allows
    # elements to be swapped in 1 position only
    @test all(in.(diff(ordered_indices), [[1, -1, 2]]))

end

end # module TestQuantLogUniform
true
