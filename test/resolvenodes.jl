module TestResolve

using Test
using TreeParzen
import TreeParzen: Delayed, Resolve, Trials

@testset "Resolve.node(::Param)" begin

    param = Dict{Symbol, Delayed.AbstractDelayed}(:b => HP.Normal(:a, 1.0, 2.0))

    # example to catch the key is present
    vals = Trials.ValsDict(:a => 1)
    @test_throws KeyError Resolve.node(param, vals)
    # example when the key is not present
    val = Trials.ValsDict(:c => 1)
    @test haskey(Resolve.node(param, val), :b)

end

end
