module TestRandInt

using Test
using TreeParzen
import TreeParzen: Delayed, API, HP, Config, ask, tell!

const poly_objective = params -> Float64(abs(evalpoly(params[:t], [0, -1500, 680, -63, 1])))

@testset "Delayed.randint overloads" begin
    @test Delayed.randint(5) in 0:4
    @test Delayed.randint(5, 0) == Int[]
    @test length(Delayed.randint(5, 3)) == 3
    @test length(Delayed.randint([5, 4, 3], 3)) == 3
    @test length(Delayed.randint([5, 4, 3], [3])) == 3

    sample = Delayed.randint([1, 2, 5], 3)
    @test sample[1] == 0
    @test sample[2] in 0:1
    @test sample[3] in 0:4

    @test_throws ArgumentError Delayed.randint(0)
    @test_throws ArgumentError Delayed.randint(0, 1)
    @test_throws ArgumentError Delayed.RandInt(0)
    @test_throws ArgumentError Delayed.randint([5, 4, 3], 2)
    @test_throws ArgumentError Delayed.randint([5, 4, 3], [3, 1])
end

@testset "Samplers.randint" begin
    config = Config()

    post, probabilities = TreeParzen.Samplers.randint(Int[], 4, 200, config)
    @test length(post) == 200
    @test all((0 .<= post) .& (post .< 4))
    @test sum(probabilities) ≈ 1.0

    post, probabilities = TreeParzen.Samplers.randint([0, 0, 1, 3], 4, 100, config)
    @test length(post) == 100
    @test all((0 .<= post) .& (post .< 4))
    @test sum(probabilities) ≈ 1.0
end

true

@testset "HP.RandInt integration" begin
    space = Dict(:x => HP.RandInt(:x, 6))

    trial = ask(space)
    @test trial.hyperparams[:x] in 0:5

    config = Config(random_trials = 5)
    trials = TreeParzen.Trials.Trial[]
    for i in 1:10
        t = ask(space)
        tell!(trials, t, Float64(i))
    end

    tp_trial = ask(space, trials, config)
    @test tp_trial.hyperparams[:x] in 0:5
end

@testset "Distribution uniformity (single-arg randint)" begin
    # Adapted from hyperopt test_basic: test that randint(upper) samples are roughly uniform
    space = Dict(:a => HP.RandInt(:a, 5))
    x = zeros(Int, 5)

    for i in 1:1000
        trial = ask(space)
        sample = trial.hyperparams[:a]
        x[sample + 1] += 1  # Convert 0-indexed to 1-indexed for array
    end

    # Each bucket should have roughly 200 ± 100 samples (100 < count < 300)
    for i in x
        @test 100 < i < 300
    end
end

@testset "Distribution uniformity (larger randint)" begin
    # Test with larger range to verify uniform distribution
    space = Dict(:a => HP.RandInt(:a, 15))
    x = zeros(Int, 15)

    for i in 1:1000
        trial = ask(space)
        sample = trial.hyperparams[:a]
        x[sample + 1] += 1
    end

    # For 15 buckets with 1000 samples, expect ~67 per bucket
    # Allow range of 30-100 per bucket
    for i in x
        @test 30 < i < 100
    end
end

@testset "Random search with randint space" begin
    # Adapted from hyperopt TestSimpleFMin.test_random_runs
    space = Dict(:t => HP.RandInt(:t, 100))
    best = API.fmin(
        poly_objective,
        space,
        150;
        random_trials = 150,
    )

    @test best[:t] in 0:99
end

@testset "TPE search with randint space" begin
    # Adapted from hyperopt TestSimpleFMin.test_tpe_runs
    space = Dict(:t => HP.RandInt(:t, 100))
    best = API.fmin(
        poly_objective,
        space,
        100;
        random_trials = 10,
    )

    @test best[:t] in 0:99
end

@testset "TPE finds minimum within range" begin
    # Test that TPE correctly optimizes with randint space
    # Objective with known minimum at value 50
    objective(params) = Float64((params[:t] - 50)^2)

    space = Dict(:t => HP.RandInt(:t, 100))
    best = API.fmin(
        objective,
        space,
        200;
        random_trials = 20,
    )

    @test best[:t] in 0:99
    @test objective(best) <= 4
end

end

true