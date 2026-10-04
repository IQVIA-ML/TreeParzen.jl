"""
Based on
https://hyperopt.github.io/hyperopt/
"""
module TestQuantNormal

using Test
using TreeParzen

# Simple quantnormal
# Called HP.QuantNormal multiple number of times to check if the distribution of values
# matched the expected given input parameters and also a test that the values appear at
# correct intervals
d = []
for i in 1:10000
    a = HP.QuantNormal(:x, 0.0, 1.0, 2.0)
    b = TreeParzen.Resolve.node(a, TreeParzen.Trials.ValsDict())
    push!(d, b)
end
@test all(in.(d, [[-4.0, -2.0, 0, 2.0, 4.0]]))

@testset "adaptive quantnormal posterior sampler" begin
    obs = TreeParzen.ConstrainedVectors.Observations([-1.0, 0.0, 1.0])
    q = 0.5
    draws, mixture = TreeParzen.Samplers.quantnormal(
        obs, 0.0, 1.0, q, 500, Config()
    )

    @test length(draws) == 500
    @test all(isapprox.(draws.v ./ q, round.(draws.v ./ q); atol=1e-10))
    @test sum(mixture.weights) ≈ 1.0
end

end # module TestQuantNormal
true
