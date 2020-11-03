module API

using DocStringExtensions

using ..Configuration
import ..Delayed
import ..Graph
import ..Resolve
import ..Trials
import ..Types

export ask
export fmin
export provide_recommendation
export tell!

"""
$(TYPEDSIGNATURES)
Provides a suggestion based on random search to generate hyperparameter values.
Also can generate trials to be evaluated from Dict of points.
"""
function ask(space::Types.SPACE_TYPE)::Trials.Trial

    Graph.checkspace(space)

    vals = Trials.ValsDict()
    hyperparams = Resolve.node(space, vals)

    return Trials.Trial(hyperparams, vals, Inf)
end
"""
$(TYPEDSIGNATURES)
Provides a suggestion based on tree-parzen estimation
"""
function ask(space::Types.SPACE_TYPE, trials::Vector{Trials.Trial}, config::Config)::Trials.Trial

    Graph.checkspace(space)

    # Run a few initial random jobs before doing tree-parzen.
    if length(trials) < config.random_trials
        return ask(space)
    end

    # Get results from previous iterations and resolve posterior distributions
    vals, hyperparams = Resolve.node(space, trials, config)

    return Trials.Trial(hyperparams, vals, Inf)
end

"""
$(TYPEDSIGNATURES)
Stores the evaluation of a trial and composes a trial history
"""
function tell!(trials::Vector{Trials.Trial}, trial::Trials.Trial, loss::Float64)::Nothing
    tell!(trial, loss)
    push!(trials, trial)

    return nothing
end
"""
$(TYPEDSIGNATURES)

Stores the evaluation of a trial only
"""
function tell!(trial::Trials.Trial, loss::Float64)::Nothing
    trial.loss = loss
    return nothing
end

"""
$(TYPEDSIGNATURES)
Gets us the "best" parameter combination
"""
function provide_recommendation(trials::Vector{Trials.Trial})
    losses = [trial.loss for trial in trials]

    return trials[argmin(losses)].hyperparams
end

check_rval(rval::Float64, fn_symbol::Symbol) = nothing
function check_rval(rval::Any, fn_symbol::Symbol)
    throw(TypeError(
        fn_symbol, "The function you submitted to TreeParzen didn't return a Float",
        Float64, rval
    ))
end

#### Below functions are to support fmin
"""
$(TYPEDSIGNATURES)

Apply the `hyperparams` to the `fn` and check that the result is `Float64`.
"""
function evaluate_hyperparams(fn::Function, hyperparams::Dict{Symbol, T} where T)::Float64

    rval = try
        fn(hyperparams)
    catch error
        throw(ArgumentError(string(
            "The function called ", fn, " that you submitted to fmin could not be run.",
            " The error message was: ", error)
        ))
    end
    check_rval(rval, Symbol(fn))

    return rval
end

"""
$(TYPEDSIGNATURES)

# Arguments

- `points` : A list of trials chosen by the user to be evaluated before optimisation.
    The list can be empty.
- `fn` : The user's function to be minimised.
- `space` : A `Dict` containing objects from the `TreeParzen.HP` module that describe the
    space of hyperparameters that TreeParzen is allowed to search.
- `N` : Number of trials to perform.
- `config` : A `Config` object.
- `logging_interval` : At what interval the update of trials should be logged. The default
    value is `-1`, where only the final total will be logged. If the parameter is specified
    you get interval log statements of iterations.

Example:

```julia
    API.run(
        Trials.Trial[], x -> x[:loss],
        space,
        1000,
        config;
        logging_interval = 100
    )
```
   Output:
    [ Info: fmin: 100 / 1000 trials carried out
    [ Info: fmin: 200 / 1000 trials carried out
    [ Info: fmin: 300 / 1000 trials carried out
    [ Info: fmin: 400 / 1000 trials carried out
    [ Info: fmin: 500 / 1000 trials carried out
    [ Info: fmin: 600 / 1000 trials carried out
    [ Info: fmin: 700 / 1000 trials carried out
    [ Info: fmin: 800 / 1000 trials carried out
    [ Info: fmin: 900 / 1000 trials carried out
    [ Info: fmin: 1000 / 1000 trials carried out

"""
function run(
    points::Vector{Trials.Trial}, fn::Function, space::Types.SPACE_TYPE, N::Int,
    config::Config;
    logging_interval::Int = -1,
)::Vector{Trials.Trial}

    trials = Trials.Trial[]

    if logging_interval == -1
        logging_interval = N
    end

    if logging_interval > N
        @warn("The logging_interval ($logging_interval) given is higher than total number of requested trials ($N)")
    end

    if length(trials) > 0
        @info("fmin: $(length(trials)) / $(N) trials carried out")
    end

    # If any points to evaluate have been added, do them first
    for point in points
        loss = evaluate_hyperparams(fn, point.hyperparams)
        API.tell!(trials, point, loss)
    end

    for i in (length(trials) + 1):N
        new_trial = API.ask(space, trials, config)
        loss = evaluate_hyperparams(fn, new_trial.hyperparams)
        API.tell!(trials, new_trial, loss)
        if (i % logging_interval) == 0
            @info("fmin: $(i) / $(N) trials carried out")
        end
    end

    return trials
end

"""
$(TYPEDSIGNATURES)

Find the set of hyperparameters that return the lowest value from the submitted function.

# Arguments

- `fn` : The function that you want to evaluate. Note that the function needs to accept a
    single `Dict` and return a `Float64`. If your function does not accept a `Dict` you will
    need to create a wrapping function that translates the content of the `Dict` to the
    parameters your function needs. Similarly if you want to maximise your function instead
    of minimise it, your wrapping function will need to invert the output.
    See the TreeParzen user guide for examples.
- `space` : A `Dict` containing objects from the `TreeParzen.HP` module that describe the
    space of hyperparameters that TreeParzen is allowed to search.
- `N` : Number of trials to perform.
- `points` : A list of trials chosen by the user to be evaluated before optimisation. The
    list can be empty. Example:
        `[Dict(:x => 0.0, :y => 0.0), Dict(:x => 1.0, :y => 2.0)]`

## Optional keyword arguments

- `threshold::Float64` : A value between `0` and `1`, which controls the probability threshold at
    which expected improvement criteria is modeled.
- `linear_forgetting::Int` : A positive value which controls the number of historic points which
    are used for probabilistic modelling, and older points beyond this are linearly
    de-weighted.
- `draws::Int` : A positive value which controls the number of samples to draw when making a
    recommendation for next optimisation candidate.
- `random_trials::Int` : A positive value which controls the number of trials of randomly
    generated candidate points before TreeParzen optimisation is used.
- `prior_weight::Float64` : A value between `0` and `1`, which controls importance of user specified
    probabilistic parameters vs the history of trials.
- `logging_interval::Int` : An integer specifying after how many iterations should a progress
    statement be logged out. Default, -1, will log only upon completion.
"""
function fmin(
    fn::Function, space::Types.SPACE_TYPE, N::Int, points::Vector{Trials.Trial};
    threshold::Float64 = 0.25, linear_forgetting::Int = 25, draws::Int = 24,
    random_trials::Int = 20, prior_weight::Float64 = 1.0, logging_interval::Int = -1,
)
    Graph.checkspace(space)

    config = Config(threshold, linear_forgetting, draws, random_trials, prior_weight)

    # Evaluate the trials
    trials = run(points, fn, space, N, config; logging_interval = logging_interval)

    @info("Successfully completed fmin ")

    return API.provide_recommendation(trials)
end
function fmin(
    fn::Function, space::Types.SPACE_TYPE, N::Int,
    points::Vector{<: Types.SPACE_TYPE}; kwargs...
)
    if N < length(points)
        throw(ArgumentError(string(
            "You have asked for fewer steps than the number of points to evaluate: N ", N,
            "points ", length(points)
        )))
    end

    return fmin(fn, space, N, API.ask.(points); kwargs...)
end
function fmin(fn::Function, space::Types.SPACE_TYPE, N::Int; kwargs...)
    return fmin(fn, space, N, Trials.Trial[]; kwargs...)
end

end  # module API
