"""
TreeParzen.MLJTreeParzen

Submodule containing TreeParzen.jl interface overrides for MLJTuning.jl

Objects:
- `MLJTreeParzenSpace`
- `MLJTreeParzenTuning`


MLJTuning methods overridden for `MLJTreeParzenTuning` object:
- `default_n`
- `result`
- `models!`
- `setup`

Example usage:
```julia
# ] add MLJBase MLJModels MLJTuning DecisionTree

using MLJBase, MLJModels, MLJTuning, TreeParzen
X, y = @load_iris

@load DecisionTreeClassifier

space = (Dict(
    :min_purity_increase => HP.Uniform(:min_purity_increase, 0.0, 1.0),
    :merge_purity_threshold => HP.Uniform(:merge_purity_threshold, 0.0, 1.0),
    :pdf_smoothing => HP.Uniform(:pdf_smoothing, 0.0, 1.0),
))

dtc = DecisionTreeClassifier()

tm = TunedModel(
    model=dtc,
    ranges=space,
    tuning=MLJTreeParzen.MLJTreeParzenTuning(),
    n=250,
    resampling=CV(nfolds=3),
    measure=cross_entropy
)

mach = machine(tm, X, y)
fit!(mach)

best_model = fitted_params(mach).best_model
@show(best_model.min_purity_increase)
@show(best_model.merge_purity_threshold)
@show(best_model.pdf_smoothing)
```
"""
module MLJTreeParzen

using DocStringExtensions

import MLJTuning

using ..API
using ..Configuration
import ..Graph
import ..Trials

######################################
# Custom data structures and methods #
######################################

"""
$(TYPEDEF)

Data structure to store the user selected search space plus any user-provided suggestions.
There are multiple constructors to support a variety of use cases.

$(TYPEDFIELDS)
"""
struct MLJTreeParzenSpace
    # deliberately not mutable, although bear in mind the underlying datatypes are ...
    """
    The user's space as a dictionary of parameter names and (potentially nested) `TreeParzen.HP.*` expressions
    """
    space::Dict{Symbol}
    """
    The set of fixed points suggested provided to search.
    The number of random warn-up rounds is reduced by the amount of suggestions provided
    """
    suggestions::Vector{Dict{Symbol}}
end

"""
$(TYPEDSIGNATURES)

Construct a hyperparameter search space from a `input_space` with keys as parameter names
and values as `TreeParzen.HP.*` tunable distributions (or composite expressions of these).

Example:
```julia
using TreeParzen

search = MLJTreeParzen.MLJTreeParzenSpace(
    Dict(
        :num_layers => HP.Choice(:num_layers, [2, 3, 4]),
        :num_nodes => HP.QuantUniform(:num_nodes, 20.0, 50.0, 1.0),
    )
);
```
"""
MLJTreeParzenSpace(input_space::Dict{Symbol}) = MLJTreeParzenSpace(
    Graph.checkspace(input_space), Dict{Symbol}[]
)
"""
$(TYPEDSIGNATURES)

As above, but also accepting `suggestion` with parameter names and specific values to evaluate
the function to be tuned with.

The default constructor accepts a vector of suggestions.

Example (single suggestion):
```julia
using TreeParzen

search = MLJTreeParzen.MLJTreeParzenSpace(
    Dict(
        :num_layers => HP.Choice(:num_layers, [2, 3, 4]),
        :num_nodes => HP.QuantUniform(:num_nodes, 20.0, 50.0, 1.0),
    ),
    Dict(:num_layers => 3, :num_nodes => 40.0)
);
```

Example (multiple suggestions):
```julia
using TreeParzen

search = MLJTreeParzen.MLJTreeParzenSpace(
    Dict(
        :num_layers => HP.Choice(:num_layers, [2, 3, 4]),
        :num_nodes => HP.QuantUniform(:num_nodes, 20.0, 50.0, 1.0),
    ),
    [
        Dict(:num_layers => 3, :num_nodes => 40.0),
        Dict(:num_layers => 3, :num_nodes => 30.0),
    ]
);
```

"""
MLJTreeParzenSpace(input_space::Dict{Symbol}, suggestion::Dict{Symbol}) = MLJTreeParzenSpace(
    Graph.checkspace(input_space), [suggestion]
)


"""

$(TYPEDEF)

The TreeParzen MLJ tuning object to be passed to MLJTuning APIs.

"""
struct MLJTreeParzenTuning <: MLJTuning.TuningStrategy
    config::Config
    max_simultaneous_draws::Int
end

"""
$(TYPEDSIGNATURES)

Keyword argument constructor
- `threshold::Float64`; default 0.25. This is the same as `TreeParzen.Config.gamma`
- `draws::Int`; default 24. This is the same as `TreeParzen.Config.draws`
- `linear_forgetting::Int`; default 25. This is the same as `TreeParzen.Config.linear_forgetting`
- `prior_weight::Float64`; default 1.0. This is the same as `TreeParzen.Config.prior_weight`
- `random_trials::Int`; default 20. This is the same as `TreeParzen.Config.random_trials`
- `max_simultaneous_draws::Int`; default 1. MLJ Interface specific, controls the number of
    candidate hyperparameters drawn without history updates during TreeParzen optimisation
    (not during suggestions/random sampling warmup), to enable MLJ parallel model training.
"""
function MLJTreeParzenTuning(;
    threshold::Float64 = 0.25,
    draws::Int = 24,
    linear_forgetting::Int = 25,
    prior_weight::Float64 = 1.0,
    random_trials::Int = 20,
    max_simultaneous_draws::Int = 1, # this only applies to TreeParzen draws, not suggestions or random
)

    config = Config(threshold, linear_forgetting, draws, random_trials, prior_weight)

    # hardcode simultaneous param to 1 for now
    return MLJTreeParzenTuning(config, max_simultaneous_draws)

end


################################################
# Start of MLJTuning interface implementations #
################################################


MLJTuning.default_n(tuning::MLJTreeParzenTuning, range) = 50


function MLJTuning.result(tuning::MLJTreeParzenTuning, history, state, evaluation, trial_in_progress)

    completed_trial = deepcopy(trial_in_progress)
    tell!(completed_trial, first(evaluation.measurement)) # `best` references the first element, which is a bit scary
    # doing like this means we can keep the existing implementation for `best`
    return (measure=evaluation.measure, measurement=evaluation.measurement, trial_object=completed_trial)

end


# this is the bit where we generate the internal history from the MLJ history
# where we have injected the trial object ourselves using a modified `result` call
# history is a vector of tuple of (model, result), where result is a namedtuple
get_trialhist(history::Nothing) = Trials.Trial[]
get_trialhist(history) = getindex.(getindex.(history, 2), :trial_object)

histlen(history::Nothing) = 0
histlen(history) = length(history)

function MLJTuning.models!(
    strategy::MLJTreeParzenTuning,
    model,
    history,
    state::MLJTreeParzenSpace,
    remaining,
    verbosity,
)::Vector{Tuple{Any, Trials.Trial}}

    num_hist = histlen(history)
    num_suggest = num_hist == 0 ? length(state.suggestions) : 0
    modeltype = typeof(model)
    max_draws = num_hist == 0 ? strategy.config.random_trials - num_suggest : strategy.max_simultaneous_draws
    trialhist = get_trialhist(history)

    # we could handle this but logic more complex and also it makes limited sense to permit anyway
    if max_draws < 0
        throw(ArgumentError("Number of suggestions cannot be more than number of warmup jobs: $num_suggest > $(strategy.config.random_trials)"))
    end

    candidates = Trials.Trial[]

    # we only ever do this when num_hist == 0
    for i in 1:num_suggest
        push!(candidates, ask(state.suggestions[i]))
    end

    # when num_hist == 0 this is the full number of random (independent draws) (minus suggestions as drawn above)
    # else its the max number of times we can ask TP from its distributions
    for i in 1:max_draws
        push!(candidates, ask(state.space, trialhist, strategy.config))
    end

    return [
        (modeltype(; candidate.hyperparams...), candidate)
            for candidate in candidates
    ]

end

# let the user construct the MLJTreeParzenSpace object directly or just specify a space dict, which will do the construction
MLJTuning.setup(tuning::MLJTreeParzenTuning, model, range::Dict{Symbol}, verbosity) = MLJTreeParzenSpace(range)
MLJTuning.setup(tuning::MLJTreeParzenTuning, model, space::MLJTreeParzenSpace, verbosity) = space

end # module
