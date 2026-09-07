module TestConstrainedVectors

using Test
using TreeParzen

CV = TreeParzen.ConstrainedVectors
IO = TreeParzen.IndexObjects

@testset "IndexVector conversion and validation" begin
    @test IO.IndexVector(Real[1, 2]).v == [1, 2]
    @test_throws InexactError IO.IndexVector(Real[1.5])
    @test_throws ArgumentError IO.IndexVector(Real[0])
end

@testset "ConstrainedVectors constructors" begin
    @testset "Observations / PosteriorDraws" begin
        values = [1.0, 2.0]
        obs = CV.Observations(values)
        @test length(obs) == 2
        @test obs[1] == 1.0
        @test collect(obs) == [1.0, 2.0]
        values[1] = 99.0
        @test obs[1] == 1.0

        draws = CV.PosteriorDraws([0.0, 1.0, 2.0])
        @test length(draws) == 3
        @test minimum(draws) == 0.0
        @test maximum(draws) == 2.0
    end

    @testset "LogPosteriorDraws shape and ownership" begin
        @test isempty(CV.LogPosteriorDraws())
        @test size(CV.LogPosteriorDraws()) == (0, 1)

        values = reshape([0.0, 1.0], :, 1)
        draws = CV.LogPosteriorDraws(values)
        values[1] = 99.0
        @test draws[1, 1] == 0.0
        @test_throws DimensionMismatch CV.LogPosteriorDraws(ones(2, 2))
    end

    @testset "ObsWeights bounds" begin
        values = [0.0, 0.5, 1.0]
        weights = CV.ObsWeights(values)
        values[1] = 99.0
        @test weights.v == [0.0, 0.5, 1.0]
        @test_throws DomainError CV.ObsWeights([-0.1, 1.0])
        @test_throws DomainError CV.ObsWeights([0.0, 1.1])
    end

    @testset "Probabilities simplex" begin
        p = CV.Probabilities([0.2, 0.3, 0.5])
        @test length(p) == 3
        @test p[3] == 0.5
        @test_throws DomainError CV.Probabilities([0.5, 0.6])
        @test_throws DomainError CV.Probabilities([-0.1, 1.1])
        @test isempty(CV.Probabilities(Float64[]))
    end

    @testset "ObsPair distinguishes buckets" begin
        pair = CV.ObsPair(CV.Observations([1.0]), CV.Observations([9.0]))
        @test pair.below[1] == 1.0
        @test pair.above[1] == 9.0
    end

    @testset "DistDetails requires positive sigmas" begin
        @test_throws DomainError TreeParzen.GMM.DistDetails([1.0], [0.0], [0.0])
        @test_throws DomainError TreeParzen.GMM.DistDetails([0.5, 0.5], [0.0, 1.0], [1.0, -1.0])
    end
end

end # module
true
