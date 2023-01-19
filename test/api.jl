module TestAPI

using Test
using TreeParzen
using Statistics

# The following three cases are testing two different formats of the Space (Dictionary and Vector)
# with different types of hyperparamert functions at a high lever, relating to the issue detailed 
# here https://github.com/IQVIA-ML/TreeParzen.jl/issues/86
@testset "High level test - dictionary space of choice functions test" begin
    a_list = [10, 14, 19]
    b_list = [1, 4, 9]
    Dict_space = Dict(
        :a => HP.Choice(:a, a_list),
        :b => HP.Choice(:b, b_list)
    )

    n_samples = 100
    n_random = 100

    # In order to get the best results, the number of random trails is increased
    # from 20 to 100 here and linear_forgetting is increased to 200.
    config = TreeParzen.Config(;linear_forgetting=n_samples + n_random, draws=50, threshold=0.1, random_trials=n_random)
    posterior_start = config.random_trials + 1

    trials = TreeParzen.Trials.Trial[]
    for i in 1:n_samples+n_random
        trial = ask(Dict_space, trials, config)
        tell!(trials, trial, trial.hyperparams[:b]/trial.hyperparams[:a])
    end
        
    posterior_trials = trials[posterior_start:end]
    samples_a = getindex.(getfield.(posterior_trials, Ref(:hyperparams)), Ref(:a))
    samples_b = getindex.(getfield.(posterior_trials, Ref(:hyperparams)), Ref(:b))
    vals_a = getindex.(getfield.(posterior_trials, Ref(:vals)), Ref(:a))
    vals_b = getindex.(getfield.(posterior_trials, Ref(:vals)), Ref(:b))
    same_indices_pct = mean(vals_a .== vals_b) * 100
    # With this example space and loss function, expected vals for the best param should be mostly different
    # but with the issue mentioned above, they are mostly equal due to same nid, 
    # thus the possibility that they are equal is larger than 40%
    @test same_indices_pct <= 40

    # To have the smaller loss, the expected best hyperparams[:a] is the largest value of a
    # and expected best hyperparams[:b] is the smallest value of b
    expected_a = a_list[end]
    expected_b = b_list[1]
    @test (mean(samples_b .== expected_b) * 100) >= 50 &&  (mean(samples_a .== expected_a) * 100) >= 50
end

@testset "High level test - Vector space of uniform functions test" begin

    SameQUniform_space = Dict{Symbol, Any}(
        :e => TreeParzen.HP.QuantUniform(
            :e, 1., 10., 1.0),
        :f => TreeParzen.HP.QuantUniform(
            :f, 1., 10., 1.0),
    )
    choice = [
        Dict{Symbol, Any}(:c => false, :d => TreeParzen.HP.QuantUniform(:d, 1., 10., 1.)),
        Dict{Symbol, Any}(:c => true),
    ]
    Vector_space = [
        SameQUniform_space,
        TreeParzen.HP.Choice(:cd_choice, choice),
    ]

    n_samples = 100
    n_random = 100

    # In order to get the best results, the number of random trails is increased
    # from 20 to 100 here and linear_forgetting is increased to 200.
    config = TreeParzen.Config(;linear_forgetting=n_samples + n_random, draws=50, threshold=0.1, random_trials=n_random)
    posterior_start = config.random_trials + 1

    trials = TreeParzen.Trials.Trial[]
    for i in 1:n_samples+n_random
        trial = ask(Vector_space, trials, config)
        tell!(trials, trial, trial.hyperparams[1][:f]/trial.hyperparams[1][:e])
    end

    posterior_trials = trials[posterior_start:end]
    samples_e = getindex.(getindex.(getfield.(posterior_trials, Ref(:hyperparams)), Ref(1)),Ref(:e))
    samples_f = getindex.(getindex.(getfield.(posterior_trials, Ref(:hyperparams)), Ref(1)),Ref(:f))
    vals_e = getindex.(getfield.(posterior_trials, Ref(:vals)), Ref(:e))
    vals_f = getindex.(getfield.(posterior_trials, Ref(:vals)), Ref(:f))
    same_indices_pct = mean(vals_e .== vals_f) * 100
    # With this example space and loss function, expected vals for the best param should be mostly different
    # but with the issue mentioned above, they are mostly equal due to same nid, 
    # thus the possibility that they are equal is larger than 40%.
    @test same_indices_pct <= 40

    # To have the smaller loss, the expected best hyperparams[:e] is larger than 6
    # and expected best hyperparams[:f] is smaller than 4
    expected_e = 6
    expected_f = 4
    @test (mean(samples_f .<= expected_f) * 100) >= 50 &&  (mean(samples_e .>= expected_e) * 100) >= 50
end

@testset "High level test - Vector space of internal functions test" begin

    SameInterFun_g = TreeParzen.Delayed.UnaryOperator(
        3 ^ TreeParzen.HP.QuantUniform(:g, 0., 9., (15/19)), round
    )
    SameInterFun_h = TreeParzen.Delayed.UnaryOperator(
        10 * (2 ^ TreeParzen.HP.QuantUniform(:h, 0., 9., (15/19))), round
    )

    InternalFunction_space = Dict{Symbol, Any}(
        :g => SameInterFun_g,
        :h => SameInterFun_h,
    )

    Vector_space = [InternalFunction_space]

    n_samples = 100
    n_random = 100

    # In order to get the best results, the number of random trails is increased
    # from 20 to 100 here and linear_forgetting is increased to 200.
    config = TreeParzen.Config(;linear_forgetting=n_samples + n_random, draws=50, threshold=0.1, random_trials=n_random)
    posterior_start = config.random_trials + 1

    trials = TreeParzen.Trials.Trial[]
    for i in 1:n_samples+n_random
        trial = ask(Vector_space, trials, config)
        tell!(trials, trial, trial.hyperparams[1][:h]/trial.hyperparams[1][:g])
    end

    posterior_trials = trials[posterior_start:end]
    samples_g = getindex.(getindex.(getfield.(posterior_trials, Ref(:hyperparams)), Ref(1)),Ref(:g))
    samples_h = getindex.(getindex.(getfield.(posterior_trials, Ref(:hyperparams)), Ref(1)),Ref(:h))
    vals_g = getindex.(getfield.(posterior_trials, Ref(:vals)), Ref(:g))
    vals_h = getindex.(getfield.(posterior_trials, Ref(:vals)), Ref(:h))
    same_indices_pct = mean(vals_g .== vals_h) * 100
    # With this example space and loss function, expected vals for the best param should be mostly different
    # but with the issue mentioned above, they are mostly equal due to same nid, 
    # thus the possibility that they are equal is larger than 40%.
    @test same_indices_pct <= 40
    
    # To have the smaller loss, the expected best hyperparams[:g] is larger than 3^5
    # and expected best hyperparams[:h] is smaller than 10*(2^3)
    expected_g = 3^5
    expected_h = 10*(2^3)
    @test (mean(samples_h .<= expected_h) * 100) >= 50 &&  (mean(samples_g .>= expected_g) * 100) >= 50
end

# Test ask() with a suggestion based on random search
space = Dict(
    :u => HP.Uniform(:param1, 0.0, 5.0),
    :qu => HP.QuantUniform(:param2, 0.0, 5.0, 1.0),
    :lu => HP.LogUniform(:param3, 1.0, 10.0),
    :qn => HP.QuantNormal(:param4, 2.0, 0.25, 1.0),
)

trials = [ask(space) for i in 1:1000]
@test length(trials) == 1000
@test all(typeof.(trials) .== TreeParzen.Trials.Trial)
samples_u = getindex.(getproperty.(trials, :hyperparams), :u)
samples_qu = getindex.(getproperty.(trials, :hyperparams), :qu)
samples_lu = getindex.(getproperty.(trials, :hyperparams), :lu)
samples_qn = getindex.(getproperty.(trials, :hyperparams), :qn)

@test all(0.0 .<= samples_u .<= 5.0)
@test all(in.(samples_qu, [(0, 1, 2, 3, 4, 5)]))
@test all(exp(1).<= samples_lu .<= exp(10))
# Given the parameters provided to HP.QuantNormal
# with 99.994% of the poplulation is within the given range of (1,2,3)
# so that we expect all of the samples to be in that range
@test all(in.(samples_qn, [(1, 2, 3)]))

# Test ask() with a suggestion based on Tree-Parzen estimation

# Config with threshold set up to 0.1 and linear forgetting at 25
# so that the top suggestions should fall above 9 for the given space
# when using tree-parzen estimation also limited the number of random_trials
# so that it is clear that the TPE recommendation is ran in the given examples
config = Config(0.1, 25, 24, 1, 1.0)

space_tpe = Dict(:u => HP.Uniform(:param1, 0.0, 10.0))

# ask() with random 10 trials
random_ask_trials = [ask(space_tpe) for i in 1:1000]

# example with losses set up to clearly prioritise higher hyperparam values
t_vector = TreeParzen.Trials.Trial[]
for entry in random_ask_trials
    t_entry = deepcopy(entry)
    tell!(t_entry, 10 - t_entry.hyperparams[:u])
    push!(t_vector, t_entry)
end

# example with losses set up to clearly de-prioritise higher hyperparam values
t_vector2 = TreeParzen.Trials.Trial[]
for entry in random_ask_trials
    t_entry = deepcopy(entry)
    tell!(t_entry, 10 + t_entry.hyperparams[:u])
    push!(t_vector2, t_entry)
end

# ask() using TPE suggestion
tpe_ask_trials = [ask(space_tpe, t_vector, config) for i in 1:1000]
tpe_ask_trials2 = [ask(space_tpe, t_vector2, config) for i in 1:1000]

samples_u_random = getindex.(getproperty.(random_ask_trials, :hyperparams), :u)
samples_u_tpe = getindex.(getproperty.(tpe_ask_trials, :hyperparams), :u)
samples_u_tpe2 = getindex.(getproperty.(tpe_ask_trials2, :hyperparams), :u)

# as there will always be a case when the TPE suggestion are outside the threshold,
# this test checks that the occurences of values in the top 10% range
# in the TPE suggestion is greater than the number of these higher values
# in the random search as the significant majority of TPE suggestions should be in this top range
@test length(samples_u_tpe[samples_u_tpe .>= 9]) > length(samples_u_random[samples_u_random .>= 9])
# checks that no more than 15% of samples is above 9 for random suggestion
@test length(samples_u_random[samples_u_random .>= 9])/length(samples_u_random) < 0.15
# checks that more than 50% of samples is above 9 for TPE suggestion
@test length(samples_u_tpe[samples_u_tpe .>= 9])/length(samples_u_tpe) > 0.5
# this test compares outputs of tpe suggestion with losses prioritising lower values
@test length(samples_u_tpe[samples_u_tpe .>= 9]) > length(samples_u_tpe2[samples_u_tpe2 .>= 9])
# checks that no more than 15% of samples is above 9 for TPE suggestion with reversed losses
@test length(samples_u_tpe2[samples_u_tpe2 .>= 9])/length(samples_u_tpe2) < 0.15
# checks that there are more than 50% of samples below or equal to 1 for TPE with reversed losses
@test length(samples_u_tpe2[samples_u_tpe2 .<= 1])/length(samples_u_tpe2) > 0.5

# Test tell!()
single_trial = TreeParzen.Trials.Trial(Dict(:x => 1), Dict(), 2)
tell!(single_trial, 2.7)
@test single_trial.loss == 2.7

trial_hist = [
    TreeParzen.Trials.Trial(Dict(:x => 1), Dict(), 5),
    TreeParzen.Trials.Trial(Dict(:x => 2), Dict(), 2),
]
tell!(trial_hist, deepcopy(single_trial), 7.0)
@test length(trial_hist) == 3
losses = getproperty.(trial_hist, :loss)
@test all(in.(losses, [(5.0, 2.0, 7.0)]))

# Test provide_recommendation()
vals = Dict() # values can be empty as it's not used by provide_recommendation

trial_vector = [
    TreeParzen.Trials.Trial(Dict(:x => 1), vals, 3.6),
    TreeParzen.Trials.Trial(Dict(:x => 2), vals, 3.5),
    TreeParzen.Trials.Trial(Dict(:x => 3), vals, 3.7),
]

trial_vector_multihyperparams = [
    TreeParzen.Trials.Trial(Dict(:x => 1, :y => 5), vals, 3.6),
    TreeParzen.Trials.Trial(Dict(:x => 2, :y => 8), vals, 3.7),
    TreeParzen.Trials.Trial(Dict(:x => 3, :y => 7), vals, 3.5),
    ]

recommendation = provide_recommendation(trial_vector)
@test recommendation == Dict(:x => 2)
recommendation_multi = provide_recommendation(trial_vector_multihyperparams)
@test recommendation_multi == Dict(:x => 3, :y => 7)

end #module TestAPI
true
