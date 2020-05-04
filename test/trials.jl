module TestTrials

using Test
using TreeParzen

space = Dict(:fun => HP.Uniform(:param1, 0.0, 5.0))

function_fail = Dict(
    :fun => (
        HP.Uniform(:param1, 0.0, 5.0),
        HP.Uniform(:param2, 0.0, 1.0)
    )
)
config = Config(0.25, 25, 24, 1001, 1.0)

trials = TreeParzen.API.run(TreeParzen.Trials.Trial[], x -> x[:fun], space, 3, config)

@test isa(trials[1], TreeParzen.Trials.Trial)
@test keys(trials[1].hyperparams) == keys(trials[2].hyperparams)
@test keys(trials[1].vals) == keys(trials[2].vals)

# Catches error when an invalid function returns a Tuple{Float64,Float64} where Float64 is
# expected
@test_throws TypeError TreeParzen.API.run(TreeParzen.Trials.Trial[], x ->[:fun], function_fail, 1, config)

end #module TestTrials
true
