module TestMLJUnit

using Test

import MLJTuning

using TreeParzen


struct DummyModel
    init::Bool # dummy field to initialise because we can't do DummyModel()
end
DummyModel(;kwargs...) = DummyModel(true) # we literally don't care,
                                          # we just need a constructor
mutable struct DummyGenericKwargModel{T}
    x::T
end
# many models are represented with more specific constructors such as SomeModel(; param1=1, param2=5.0), e.g. LightGBM
# there are also models represented with generic kwargs such as SomeModel(; kwargs...), e.g. EvoTrees
# the below is a testing example of a generic kwarg constructor which has been previously unsupported in TreeParzen
DummyGenericKwargModel(; x=1.0)=DummyGenericKwargModel(x)

function setup(;n_simultaneous=1,
               n_startup=20,
               space=Dict{Symbol},
               suggest=Dict{Symbol}[])

    tuning = MLJTreeParzenTuning(;max_simultaneous_draws=n_simultaneous,
                                 random_trials=n_startup)
    state = (space=MLJTreeParzen.MLJTreeParzenSpace(space, suggest),
             trialhist=TreeParzen.Trials.Trial[])

    return tuning, state
end


function complete_trials(array_of_stoof)

    trials = last.(array_of_stoof)

    for trial in trials
        TreeParzen.tell!(trial, rand(Float64))
    end

    return [(DummyModel(), (trial_object=trial,)) for trial in trials]

end


@testset "MLJ: default_n" begin

    obj = MLJTreeParzen.MLJTreeParzenTuning()
    @test MLJTuning.default_n(obj, nothing) == 50

end


@testset "MLJ: setup" begin

    tuning = MLJTreeParzen.MLJTreeParzenTuning()
    model = nothing # supposed to be a model but nobody is checking
                    # and we dont need it so lets prove it
    verbosity = nothing

    space_dict = Dict(:x => HP.Uniform(:x, 0.0, 1.0))
    suggestions = Dict(:x => 0.222)
    suggestions_list = [Dict(:x => 0.222), Dict(:x => 0.444)]
    preconstructed_space = MLJTreeParzen.MLJTreeParzenSpace(space_dict)

    preconstruted_suggest =
        MLJTreeParzen.MLJTreeParzenSpace(space_dict, suggestions)

    preconstruted_suggestions =
        MLJTreeParzen.MLJTreeParzenSpace(space_dict, suggestions_list)

    # test with just a dict space:
    state = MLJTuning.setup(tuning, model, space_dict, 123, verbosity)
    @test state.trialhist isa Vector{<:TreeParzen.Trials.Trial}
    @test state.space isa MLJTreeParzen.MLJTreeParzenSpace

    # test with preconstructed space and that we get same thing:
    state = MLJTuning.setup(tuning, model, preconstructed_space, 123, verbosity)
    @test state.space isa MLJTreeParzen.MLJTreeParzenSpace
    @test state.space == preconstructed_space

    # test with preconstructed space with a sugggestion and we get the
    # same thing:
    state = MLJTuning.setup(tuning, model, preconstruted_suggest, 123, verbosity)
    @test state.space isa MLJTreeParzen.MLJTreeParzenSpace
    @test state.space == preconstruted_suggest

    # test with preconstructed space with multiple suggestions and we
    # get the same thing:
    state = MLJTuning.setup(tuning, model, preconstruted_suggestions, 123, verbosity)
    @test state.space isa MLJTreeParzen.MLJTreeParzenSpace
    @test state.space == preconstruted_suggestions

end

@testset "MLJ: models" begin

    # scenarios to cover (or properties to check):
    # 1) with suggestions < num startup (return num startup followed by num simultaenous draw)
    # 2) with suggestions > num startup (throw exception)
    # 3) with suggestions == num startup (return num startup followed by num simultaenous draw)
    # 4) no suggestions (return num startup followed by num simultaenous draw)
    # 5) round 1 always returns num startup and all subsequent rounds returns num simultaneous draw

    space = Dict(:x => HP.Uniform(:x, -5., 5.))
    testmodel = DummyModel()
    fakehist = complete_trials([(nothing, TreeParzen.ask(space)) for i in 1:100]) # do too many, we can cut it down

    @testset "no suggestions" begin

        suggestions = Dict{Symbol}[]
        tuning, state = setup(;n_startup=3,space=space, suggest=suggestions)

        output, _ = MLJTuning.models(tuning, testmodel, nothing, state, 0, 0)

        @test output isa Vector{<:Tuple{Any, TreeParzen.Trials.Trial}}
        @test length(output) == 3

    end

    @testset "less suggestions" begin

        suggestions = [Dict(:x => 1), Dict(:x => 2)]
        tuning, state = setup(;n_startup=3,space=space, suggest=suggestions)

        output, _ = MLJTuning.models(tuning, testmodel, nothing, state, 0, 0)

        @test output isa Vector{<:Tuple{Any, TreeParzen.Trials.Trial}}
        @test length(output) == 3
        @test getindex.(getproperty.(last.(output), :hyperparams), :x)[1:2] == [1, 2]

    end

        @testset "exact suggestions" begin

        suggestions = [Dict(:x => 1), Dict(:x => 2), Dict(:x => 3)]
        tuning, state = setup(;n_startup=3,space=space, suggest=suggestions)

        output, _ = MLJTuning.models(tuning, testmodel, nothing, state, 0, 0)

        @test output isa Vector{<:Tuple{Any, TreeParzen.Trials.Trial}}
        @test length(output) == 3
        @test getindex.(getproperty.(last.(output), :hyperparams), :x) == [1, 2, 3]

    end

        @testset "too many suggestions" begin

        suggestions = [Dict(:x => 1), Dict(:x => 2), Dict(:x => 3), Dict(:x => 4)]
        tuning, state = setup(;n_startup=3,space=space, suggest=suggestions)

        @test_throws ArgumentError MLJTuning.models(tuning, testmodel, nothing, state, 0, 0)

    end

    @testset "num draws" begin

        for sim_value in 1:10

            suggestions = Dict{Symbol}[]
            tuning, state = setup(;n_simultaneous=sim_value,space=space, suggest=suggestions)

            output, _ = MLJTuning.models(tuning, testmodel, nothing, state, 0, 0)

            @test output isa Vector{<:Tuple{Any, TreeParzen.Trials.Trial}}
            @test length(output) == 20

            fakeMLJhist = map(fakehist) do entry
                (model       = entry[1],
                 measure     = [MLJTuning.rmse],
                 measurement = rand(),
                 metadata    = entry[2].trial_object)
            end

            for histlen in 21:30

                output, state = MLJTuning.models(tuning,
                                           testmodel,
                                           fakeMLJhist[1:histlen],
                                           state,
                                           0,
                                           0)
                @test length(output) == sim_value
                @test length(state.trialhist) == histlen

            end

        end

    end
    # an example of a generic kwarg constructor which has been previously unsupported in TreeParzen
    testGenericKwargModel = DummyGenericKwargModel(x=5.0)

    @testset "no suggestions with a generic kwarg model constructor example" begin

        suggestions = Dict{Symbol}[]
        tuning, state = setup(;n_startup=3,space=space, suggest=suggestions)

        output, _ = MLJTuning.models(tuning, testGenericKwargModel, nothing, state, 0, 0)

        @test output isa Vector{<:Tuple{Any, TreeParzen.Trials.Trial}}
        @test length(output) == 3

    end

end


end # module
true
