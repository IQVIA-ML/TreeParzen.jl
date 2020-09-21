# ] Gadfly@1.2.1 TreeParzen MLJ DataFrames XGBoost MLJModels MLJTuning CategoricalArrays ComputationalResources


import Gadfly, MLJ, DataFrames, MLJTuning, MLJModels, CategoricalArrays, ComputationalResources

using TreeParzen

import Random: seed!

Gadfly.set_default_plot_size(20Gadfly.cm, 16Gadfly.cm)

# this isn't great but it will do
seed!(1999)

# Helper function for turning strings into not strings,
# seems to be required for use with XGBoostRegressor,
# Ints and other things make it complain
conv(x::CategoricalArrays.CategoricalArray) = Float64.(MLJ.int(x))
conv(x) = Float64.(x)

function opt_hist_plot(tuning_history, title, path)

    metric_name = String(Symbol(first(first(tuning_history).measure)))
    metric = first.(getfield.(tuning_history, :measurement))
    cummetric = accumulate(min, metric)

    upper_bound = min(minimum(metric) * 3, maximum(metric))

    plotobj = Gadfly.plot(
        Gadfly.layer(x=1:length(metric), y=metric, Gadfly.Geom.line, Gadfly.Theme(default_color=Gadfly.colorant"orange")),
        Gadfly.layer(x=1:length(metric), y=cummetric, Gadfly.Geom.step),
        Gadfly.Guide.ylabel("MLJ Tuning optimisation measurement: $metric_name"; orientation=:vertical),
        Gadfly.Guide.xlabel("Sequence"),
        Gadfly.Coord.Cartesian(ymax=upper_bound),
        Gadfly.Guide.Title(title),
    )

    plotobj |> Gadfly.SVGJS(path)
    Gadfly.display(plotobj)
end

#### constants ####
NUM_CV_FOLDS = 4
PCT_TRAIN_DATA = 0.75
NUM_TP_ITER_SMALL = 25
NUM_TP_ITER_LARGE = 250
#### end constants ####

MLJ.@load XGBoostRegressor

Features, targets = MLJ.@load_reduced_ames
# This one turns strings into not strings, and everything into Float64s
IntCat_Features = NamedTuple{keys(Features)}(conv.(values(deepcopy(Features))))
Features = DataFrames.DataFrame(Features)
IntCat_Features = DataFrames.DataFrame(IntCat_Features)

# Do hold-out partitioning. If you want same results each time use shuffle=false or set RNG seed
train, test = MLJ.partition(eachindex(targets), PCT_TRAIN_DATA, shuffle=true)

train_features = IntCat_Features[train, :]
train_targets = targets[train]

test_features = IntCat_Features[test, :]
test_targets = targets[test]

num_train_data = length(train_targets)
training_data_per_fold = (num_train_data / NUM_CV_FOLDS) * (NUM_CV_FOLDS - 1)


# Some prior decisions (almost arbitrary, there aren't strong reasons to make these decisions)
#
# - Test holdout : 25%
# - 4 fold cross validation -- this leaves each individual training set with ~820 data points and ~200 for evaluation on each fold
# - Optimising using MAE as the metric because we will assess final quality on RMSL (and because for this example we're going to optimise directly in logspace)



# We identified 7 potentially interesting optimisation parameters for gradient boosted trees.
# One of the first things we note is that even if we just selected 2 points on the relevant ranges,
# a cartesian product (grid search) over this space would be 128 evaluations
# Search over 3 points on each axis (say, min, median, max) would take the search to well over 2000 evaluations.
space = Dict(
    :num_round => HP.QuantUniform(:num_round, 1., 500., 1.),
    :eta => HP.LogUniform(:eta, -3., 0.),
    :gamma => HP.LogUniform(:gamma, -3., 3.),
    :max_depth => HP.QuantUniform(:max_depth, 1., ceil(log2(training_data_per_fold)), 1.0),
    :min_child_weight => HP.LogUniform(:min_child_weight, -5., 2.),
    :lambda => HP.LogUniform(:lambda, -5., 2.),
    :alpha => HP.LogUniform(:alpha, -5., 2.),
)

model_tpl = XGBoostRegressor()

# Here we are tuning the model by MAE in logspace, which isn't necessarily
# right but we also need to ensure only positive valued outputs are produced.
# And this is easier for TreeParzen demonstration purposes than constructing a full
# learning network which log targets and exp predictions
tuning = MLJTuning.TunedModel(
    model=model_tpl,
    ranges=space,
    tuning=TreeParzen.MLJTreeParzen.MLJTreeParzenTuning(),
    n=NUM_TP_ITER_SMALL,
    resampling=MLJ.CV(nfolds=NUM_CV_FOLDS),
    measure=MLJ.mav,
)

mach = MLJ.machine(tuning, train_features, log.(train_targets))
MLJ.fit!(mach)


# julia> fit!(mach)
# [ Info: Training Machine{DeterministicTunedModel{MLJTreeParzenTuning,…}} @ 5…58.
# [ Info: Attempting to evaluate 25 models.
# Evaluating over 20 metamodels: 100%[=========================] Time: 0:03:21
# Evaluating over 1 metamodels: 100%[=========================] Time: 0:00:07
# Evaluating over 1 metamodels: 100%[=========================] Time: 0:00:02
# Evaluating over 1 metamodels: 100%[=========================] Time: 0:00:14
# Evaluating over 1 metamodels: 100%[=========================] Time: 0:00:05
# Evaluating over 1 metamodels: 100%[=========================] Time: 0:00:08
# Machine{DeterministicTunedModel{MLJTreeParzenTuning,…}} @ 5…58
#
# julia>



# perform the evaluation(s) -- predict for a TunedModel will use best one (or best parameters trained on whole data, depending on settings)
pred = exp.(MLJ.predict(mach, test_features))
@show MLJ.rmsl(test_targets, pred)

best_model = MLJ.fitted_params(mach).best_model

for x in keys(space) println("$x = $(getproperty(best_model, x))") end

opt_hist_plot(mach.report.history, "Simple optimisation of tree boosting", joinpath(@__DIR__, "../../../examples/simple_mlj_demo/images/simple_tree_tuning.svg"))

# To demonstrate use of suggestions, we can take the best result from last tuning run.
# BEAR IN MIND that this is cheating from a DS perspective, this is just to demonstrate the functionality.
suggestion = Dict(key => getproperty(best_model, key) for key in keys(space))

search = TreeParzen.MLJTreeParzen.MLJTreeParzenSpace(space, suggestion)

tuning = MLJTuning.TunedModel(
    model=model_tpl,
    ranges=search,
    tuning=TreeParzen.MLJTreeParzen.MLJTreeParzenTuning(;random_trials=3),
    n=NUM_TP_ITER_SMALL,
    resampling=MLJ.CV(nfolds=NUM_CV_FOLDS),
    measure=MLJ.mav,
)

mach = MLJ.machine(tuning, train_features, log.(train_targets))
MLJ.fit!(mach)

# perform the evaluation(s) -- predict for a TunedModel will use best one (or best parameters trained on whole data, depending on settings)
pred = exp.(MLJ.predict(mach, test_features))
@show MLJ.rmsl(test_targets, pred)

best_model = MLJ.fitted_params(mach).best_model

for x in keys(space) println("$x = $(getproperty(best_model, x))") end

# we can also accelerate learning by using parallelism
# notice how above we had 20 metamodels followed by sequences of 1?
# What is happening here is that there is an initial stage (by default 20)
# of models drawn entirely at random (using specified prior distributions)
# without probabilistic modelling. Once probabilistic modelling kicks in,
# we draw a suggestion and then update with result. This doesn't seem amenable
# to parallelism, and it isn't. However there is an additional parameter
# `max_simultaneous_draws` which allows the system to draw `n` samples before
# `updating the distribution. Whilst intuitively this allows parallelism, it
# also enables more exploration before drawing again according to updated distribution.
# If using this parameter, consider increasing the value of `linear_forgetting`
# from its default of 25 to a higher number -- a good place is probably at least `n*25`. The
# `linear_forgetting` parameter keeps most n recently observed results equally weighted
# and older observations are  weighted by a linear ramp according to their age.

# grab previous best result again, hey we started cheating, might as well continue.
suggestion = Dict(key => getproperty(best_model, key) for key in keys(space))
search = TreeParzen.MLJTreeParzen.MLJTreeParzenSpace(space, suggestion)

tuning = MLJTuning.TunedModel(
    model=model_tpl,
    ranges=space,
    tuning=TreeParzen.MLJTreeParzen.MLJTreeParzenTuning(;random_trials=3, max_simultaneous_draws=2, linear_forgetting=50),
    n=NUM_TP_ITER_SMALL,
    resampling=MLJ.CV(nfolds=NUM_CV_FOLDS),
    measure=MLJ.mav,
)

mach = MLJ.machine(tuning, train_features, log.(train_targets))
MLJ.fit!(mach)

# julia> fit!(mach)
# [ Info: Training Machine{DeterministicTunedModel{MLJTreeParzenTuning,…}} @ 4…10.
# [ Info: Attempting to evaluate 25 models.
# Evaluating over 3 metamodels: 100%[=========================] Time: 0:00:35
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:07
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:00
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:50
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:12
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:24
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:23
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:03
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:16
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:03
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:23
# Evaluating over 2 metamodels: 100%[=========================] Time: 0:00:08
# Machine{DeterministicTunedModel{MLJTreeParzenTuning,…}} @ 4…10

# It's worth noting, that this changed behaviour of optimiser but didn't
# introduce parallelism directly: That's up to MLJ. So lets parallelise.
# It wouldn't really accelerate computations in this case possibly due to
# overheads, but is an option for larger tasks. Also note that we can instead
# parallelise via `resampling_acceleration` if we don't want to alter behaviour of
# optimiser, but this might be more limited.

suggestion = Dict(key => getproperty(best_model, key) for key in keys(space))
search = TreeParzen.MLJTreeParzen.MLJTreeParzenSpace(space, suggestion)

tuning = MLJTuning.TunedModel(
    model=model_tpl,
    ranges=space,
    tuning=TreeParzen.MLJTreeParzen.MLJTreeParzenTuning(;random_trials=3, max_simultaneous_draws=2, linear_forgetting=50),
    n=NUM_TP_ITER_SMALL,
    resampling=MLJ.CV(nfolds=NUM_CV_FOLDS),
    measure=MLJ.mav,
    acceleration=ComputationalResources.CPUProcesses(),
)

mach = MLJ.machine(tuning, train_features, log.(train_targets))
MLJ.fit!(mach)


# An interesting feature of note is that TreeParzen supports "tree-structured"
# parameter spaces; hence the name. Originally conceived for optimising
# Deep Belief Networks (DBNs), where parameters take form such as number of layers,
# and how many nodes are within each layer. Here, we can see the idea that if
# we have 2 layers, then number of nodes in layer 3 as a parameter is not relevant.
# We can construct a similar example using XGBoost - either we can use boosted trees
# or boosted linears. Some parameters are relevant in both cases, but even so we
# would want to model them differently, e.g. `num_iterations` might need to be lower for
# boosting trees than for boosting linear functions.

# First, create a container model (because of space nesting, we need to do this)
# plus its constructor:

mutable struct tuned_xgb <: MLJ.Deterministic
    xgb::XGBoostRegressor
end
tuned_xgb(;xgb=Dict{Symbol, Any}()) = tuned_xgb(XGBoostRegressor(;xgb...))

# quick fit and predict methods
MLJ.fit(t::tuned_xgb, verbosity::Int, X, y, w=nothing) = MLJ.fit(t.xgb, verbosity, X, y, w)
MLJ.predict(t::tuned_xgb, fitted, X) = MLJ.predict(t.xgb, fitted, X)

# Define search parameters for the tree search space
tree_space = Dict(
    :booster => "gbtree",
    :num_round => HP.QuantUniform(:num_round_tree, 50., 750., 1.),
    :eta => HP.LogUniform(:eta_tree, -3., 0.),
    :gamma => HP.LogUniform(:gamma_tree, -3., 3.),
    :max_depth => HP.QuantUniform(:max_depth_tree, 1., ceil(log2(training_data_per_fold)), 1.0),
    :min_child_weight => HP.LogUniform(:min_child_weight_tree, -5., 1.),
    :lambda => HP.LogUniform(:lambda_tree, -5., 1.),
    :alpha => HP.LogUniform(:alpha_tree, -5., 1.),
)

# Define search parameters for the linear search space
linear_space = Dict(
    :booster => "gblinear",
    :updater => "coord_descent",
    :num_round => HP.QuantUniform(:num_round_linear, 500., 1000., 1.),
    :lambda => HP.LogUniform(:lambda_linear, -10., 0.),
    :alpha => HP.LogUniform(:alpha_linear, -10., 0.),
    :feature_selector => HP.Choice(:feature_selector_linear, ["cyclic", "greedy"]),
)

# Now we combine them so that it chooses either one search space or another (and then select parameters for each)
joint_space = Dict(
    :xgb => HP.Choice(:xgb, [linear_space, tree_space])
)


# Because we have a top level conditional, crank up the number of random trials to start off with
tuning = MLJTuning.TunedModel(
    model=tuned_xgb(),
    ranges=joint_space,
    tuning=TreeParzen.MLJTreeParzen.MLJTreeParzenTuning(;random_trials=50),
    n=NUM_TP_ITER_LARGE,
    resampling=MLJ.CV(nfolds=NUM_CV_FOLDS),
    measures=MLJ.mav,
)

# Do the usual fitting
mach = MLJ.machine(tuning, train_features, log.(train_targets))
MLJ.fit!(mach)

# Print out final evaluation, but we dont know what the best model is yet
pred = exp.(MLJ.predict(mach, test_features))
@show MLJ.rmsl(test_targets, pred)

# Take a look at the best performing model
best_model = MLJ.fitted_params(mach).best_model.xgb
if best_model.booster == "gbtree"
    println("Tree params")
    for x in keys(tree_space) println("$x = $(getproperty(best_model, x))") end
else
    println("Linear params")
    for x in keys(linear_space) println("$x = $(getproperty(best_model, x))") end
end

# Look at learning history
opt_hist_plot(mach.report.history, "Conditional optimisation of tree/linear boosting", joinpath(@__DIR__, "../../../examples/simple_mlj_demo/images/joint_linear_tree_tuning.svg"))

# Find times we selected linear:
linear_selected = getfield.(getfield.(first.(mach.report.history), :xgb), :booster) .== "gblinear"
linear_hist = mach.report.history[linear_selected]

opt_hist_plot(linear_hist, "History (partial) of evaluations conditioned on linear boosts", joinpath(@__DIR__, "../../../examples/simple_mlj_demo/images/joint_linear_only_iterations.svg"))

tree_hist = mach.report.history[.!linear_selected]

opt_hist_plot(tree_hist, "History (partial) of evaluations conditioned on tree boosts", joinpath(@__DIR__, "../../../examples/simple_mlj_demo/images/joint_tree_only_iterations.svg"))
