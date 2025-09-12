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

export MLJTreeParzenTuning, MLJTreeParzenSpace

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

    # use an inner constructor to check now that checkspace returns nothing
    function MLJTreeParzenSpace(space, suggestions)
        Graph.checkspace(space)
        return new(space, suggestions)
    end

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
MLJTreeParzenSpace(input_space::Dict{Symbol}) = MLJTreeParzenSpace(input_space, Dict{Symbol}[])
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
MLJTreeParzenSpace(input_space::Dict{Symbol}, suggestion::Dict{Symbol}) = MLJTreeParzenSpace(input_space, [suggestion])


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

vector(history::Nothing) = Trials.Trial[]
vector(history::Vector) = history

get_trialhist(history) =
    map(history) do entry
        trial_object = entry.metadata
        sign = MLJTuning.signature(first(entry.measure))
        measurement = sign * first(entry.measurement)
        # @ablaom asks "Is this deepcopy really necessary?":
        completed_trial = deepcopy(trial_object)
        tell!(completed_trial, measurement)
        completed_trial
    end

update_param!(model, param, val) = hasproperty(model, param) ? setproperty!(model, param, val) : error("Invalid hyperparameter: $param")

function recursive_hyperparam_update!(model, dict)
    for (hyperparam, value) in dict
        if isa(value, Dict)
            recursive_hyperparam_update!(model, value)
        else
            update_param!(model, hyperparam, value)
        end
    end
end

function MLJTuning.models(
    strategy::MLJTreeParzenTuning,
    model,
    history,
    state,
    remaining,
    verbosity,
)

    space = state.space
    trialhist = state.trialhist

    num_hist = length(vector(history))

    # get an up-to-date the history of trial objects by appending to
    # the trial object history stored in `state`:
    recent_history =
        view(vector(history), (length(trialhist) + 1):num_hist)
    recent_trialhist = get_trialhist(recent_history)
    trialhist = vcat(state.trialhist, recent_trialhist)

    num_suggest = num_hist == 0 ? length(space.suggestions) : 0
    max_draws = num_hist == 0 ? strategy.config.random_trials - num_suggest : strategy.max_simultaneous_draws


    # we could handle this but logic more complex and also it makes limited sense to permit anyway
    if max_draws < 0
        throw(ArgumentError("Number of suggestions cannot be more than number of warmup jobs: $num_suggest > $(strategy.config.random_trials)"))
    end

    candidates = Trials.Trial[]

    # we only ever do this when num_hist == 0
    for i in 1:num_suggest
        push!(candidates, ask(space.suggestions[i]))
    end

    # when num_hist == 0 this is the full number of random (independent draws) (minus suggestions as drawn above)
    # else its the max number of times we can ask TP from its distributions
    for i in 1:max_draws
        push!(candidates, ask(space.space, trialhist, strategy.config))
    end

    newstate = (space=space, trialhist=trialhist)
    vector_of_metamodels = Tuple{Any, Trials.Trial}[]

    for candidate in candidates
        metamodel = deepcopy(model)
        recursive_hyperparam_update!(metamodel, candidate.hyperparams)
        push!(vector_of_metamodels, (metamodel, candidate))
    end

    return vector_of_metamodels, newstate

end

# let the user construct the MLJTreeParzenSpace object directly or
# just specify a space dict, which will do the construction:
MLJTuning.setup(tuning::MLJTreeParzenTuning,
                model, range::Dict{Symbol},
                n, # ignored
                verbosity) =
                    (space=MLJTreeParzenSpace(range), trialhist=Trials.Trial[])
MLJTuning.setup(tuning::MLJTreeParzenTuning,
                model,
                space::MLJTreeParzenSpace,
                n, # ignored
                verbosity) =
                    (space=space, trialhist=Trials.Trial[])

end # module
