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
        HP.Uniform(:a, 0., 1.),
        HP.Uniform(:a, 1., 2.),
        HP.Uniform(:a, 2., 3.),
    ]
)


config = Config()
trials = [ask(direct_delayed_space) for i in 1:config.random_trials]
# fill them in
[tell!(t, 1.) for t in trials]
# actually, we just wanna see that this stuff doesn't vom
newtrial = ask(direct_delayed_space, trials, config)



end
true
