module Configuration

using DocStringExtensions

export Config

"""
$(TYPEDEF)
$(TYPEDFIELDS)

- `threshold` : A value between `0` and `1`, which controls the probability threshold at
    which expected improvement criteria is modeled.
- `linear_forgetting` : A positive value which controls the number of historic points which
    are used for probabilistic modelling, and older points beyond this are linearly
    de-weighted.
- `draws` : A positive value which controls the number of samples to draw when making a
    recommendation for next optimisation canidate.
- `random_trials` : A positive value which controls the number of trials of randomly
    generated candidate points before TreeParzen optimisation is used.
- `prior_weight` : A value between `0` and `1`, which controls importance of user specified
    probabilistic parameters vs the history of trials.
"""
struct Config
    threshold::Float64
    linear_forgetting::Int
    draws::Int
    random_trials::Int
    prior_weight::Float64

    function Config(
        threshold::Float64, linear_forgetting::Int, draws::Int, random_trials::Int,
        prior_weight::Float64
    )
        if !(0 <= threshold <= 1)
            throw(ArgumentError("threshold must be between 0 and 1"))
        end
        if linear_forgetting < 0
            throw(ArgumentError("linear_forgetting must be positive"))
        end
        if draws < 0
            throw(ArgumentError("draws must be positive"))
        end
        if random_trials < 0
            throw(ArgumentError("random_trials must be positive"))
        end
        if !(0 <= prior_weight <= 1)
            throw(ArgumentError("prior_weight must be between 0 and 1"))
        end

        return new(threshold, linear_forgetting, draws, random_trials, prior_weight)
    end
end


"""
$(TYPEDSIGNATURES)

Keyword argument outer constructor
- `threshold::Float64`; default 0.25.
- `linear_forgetting::Int`; default 25.`
- `draws::Int`; default 24. `
- `random_trials::Int`; default 20.
- `prior_weight::Float64`; default 1.0.
"""
function Config(;
    threshold::Float64 = 0.25,
    linear_forgetting::Int = 25,
    draws::Int = 24,
    random_trials::Int = 20,
    prior_weight::Float64 = 1.0,
    )
    return Config(threshold, linear_forgetting, draws, random_trials, prior_weight)
end

end # module Configuration
