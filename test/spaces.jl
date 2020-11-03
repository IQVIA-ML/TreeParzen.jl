module TestSpaces

using Test
using TreeParzen


space = Dict(
    :a => HP.Choice(:a, [:a, :b, :c]),
    :b => HP.Choice(:b, ["a", "b", "c"])
)

# just asking, isn't good enough
trial = ask(space)
sample = trial.hyperparams

@test sample[:a] in (:a, :b, :c)
@test sample[:b] in ("a", "b", "c")

# gotta ask both ways
trials = TreeParzen.Trials.Trial[]
tell!(trials, trial, 2.)
trial = ask(space)
tell!(trials, trial, 2.6)

tp_trial = ask(space, trials, TreeParzen.Config())
tp_sample = tp_trial.hyperparams

# same holds true but we have to make sure we can sample from strings literals via TP asks
@test tp_sample[:a] in (:a, :b, :c)
@test tp_sample[:b] in ("a", "b", "c")



# This is to test support for using inputs other than Dicts as space definitions
direct_delayed_space = HP.Choice(:mychoice,
    [
        HP.Uniform(:a1, 0., 1.),
        HP.Uniform(:a2, 1., 2.),
        HP.Uniform(:a3, 2., 3.),
    ]
)


config = Config()
trials = [ask(direct_delayed_space) for i in 1:config.random_trials]
# fill them in
[tell!(t, 1.) for t in trials]
# actually, we just wanna see that this stuff doesn't vom
newtrial = ask(direct_delayed_space, trials, config)
# Well and check that the hyperparams is a float ....
@test newtrial.hyperparams isa Float64


direct_dict_space = HP.Choice(:mychoice,
    [
        Dict(:a => HP.Uniform(:a1, 0., 1.), :b => HP.Uniform(:b1, 0., 1.),),
        Dict(:a => HP.Uniform(:a2, 0., 1.), :b => HP.Uniform(:b2, 0., 1.),),
        Dict(:a => HP.Uniform(:a3, 0., 1.), :b => HP.Uniform(:b3, 0., 1.),),
    ]
)

trials = [ask(direct_dict_space) for i in 1:config.random_trials]
# fill them in
[tell!(t, 1.) for t in trials]
# actually, we just wanna see that this stuff doesn't vom
newtrial = ask(direct_dict_space, trials, config)
# Well and check that the hyperparams is a float ....
@test newtrial.hyperparams isa Dict
@test haskey(newtrial.hyperparams, :a)
@test haskey(newtrial.hyperparams, :b)



direct_array_space = [
    HP.Uniform(:a1, 0., 1.),
    HP.Uniform(:a2, 1., 2.),
    HP.Uniform(:a3, 2., 3.),
]


trials = [ask(direct_array_space) for i in 1:config.random_trials]
# fill them in
[tell!(t, 1.) for t in trials]
# actually, we just wanna see that this stuff doesn't vom
newtrial = ask(direct_array_space, trials, config)
# Well and check that the hyperparams is a float ....
@test newtrial.hyperparams isa Vector
@test length(newtrial.hyperparams) == 3
@test 0 <= newtrial.hyperparams[1] <= 1
@test 1 <= newtrial.hyperparams[2] <= 2
@test 2 <= newtrial.hyperparams[3] <= 3

end

true
