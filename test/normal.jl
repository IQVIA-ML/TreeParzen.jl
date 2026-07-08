module TestNormalSampler

using Statistics
using Test
using TreeParzen

@testset "Samplers.normal" begin
    config = Config()

    @testset "empty observations, prior-only mixture" begin
        mu = 0.0
        sigma = 1.0
        n = 10_000
        post, components = TreeParzen.Samplers.normal(Float64[], mu, sigma, n, config)

        @test length(post) == n
        @test all(c.sigma > 0 for c in components)
        @test sum(c.weight for c in components) ≈ 1.0
        @test all(c.weight >= 0 for c in components)
        @test isapprox(mean(post), mu; atol = 0.05)
        @test isapprox(std(post; corrected = false), sigma; rtol = 0.05)
    end

    @testset "non-empty observations" begin
        obs = Float64[-1.5, 0.0, 1.5]
        mu = 0.0
        sigma = 2.0
        n = 5_000
        post, components = TreeParzen.Samplers.normal(obs, mu, sigma, n, config)

        @test length(post) == n
        @test all(c.sigma > 0 for c in components)
        @test sum(c.weight for c in components) ≈ 1.0
        @test all(c.weight >= 0 for c in components)
        @test minimum(post) < maximum(post)
    end

    @testset "single observation (adaptive_parzen length(mus) == 1)" begin
        n = 2_000
        mu = 0.0
        sigma = 2.0
        # Prior mean to the left of the lone observation
        post_l, components_l = TreeParzen.Samplers.normal([5.0], mu, sigma, n, config)
        @test length(components_l) == 2
        @test length(post_l) == n
        @test sum(c.weight for c in components_l) ≈ 1.0
        @test all(c.sigma > 0 for c in components_l)
        @test minimum(post_l) < maximum(post_l)
        # Prior mean to the right of the lone observation (prior inserted second)
        post_r, components_r = TreeParzen.Samplers.normal([-5.0], mu, sigma, n, config)
        @test length(components_r) == 2
        @test sum(c.weight for c in components_r) ≈ 1.0
        @test all(c.sigma > 0 for c in components_r)
        @test minimum(post_r) < maximum(post_r)
    end

    @testset "linear forgetting (linear_forgetting < length(obs))" begin
        # With default linear_forgetting=25, use more than 25 observations so forgetting_weights applies
        obs = collect(range(-2.0, 2.0; length = 30))
        mu = 0.0
        sigma = 2.0
        n = 3_000
        cfg = Config(; linear_forgetting = 25)
        @test cfg.linear_forgetting < length(obs)
        post, components = TreeParzen.Samplers.normal(obs, mu, sigma, n, cfg)

        @test length(post) == n
        @test all(c.sigma > 0 for c in components)
        @test sum(c.weight for c in components) ≈ 1.0
        @test all(c.weight >= 0 for c in components)
        @test minimum(post) < maximum(post)
    end

    @testset "prior_sigma must be positive" begin
        @test_throws DimensionMismatch TreeParzen.Samplers.normal(
            Float64[], 0.0, 0.0, 10, Config(),
        )
    end
end

end

true
