module ApFilterTrials

using DocStringExtensions

using ..Configuration
import ..ConstrainedVectors: Observations, ObsPair
import ..IndexObjects
import ..Trials

function _collect_ap_filter_trials(
    nid::Symbol, trials::Vector{Trials.Trial}, config::Config,
)
    # Splitting is done this way to cope with duplicate loss values.
    # This is the number of below values that are extracted from trials by loss.
    n_below = min(Int(ceil(config.threshold * sqrt(length(trials)))), config.linear_forgetting)

    trials_by_loss = sort(trials; by = t -> t.loss)
    below = [
        trial.vals[nid]
            for trial in trials_by_loss[1:n_below]
                if nid in keys(trial.vals)
    ]
    above = [
        trial.vals[nid]
            for trial in trials_by_loss[n_below + 1:end]
                if nid in keys(trial.vals)
    ]
    return below, above
end

"""
$(TYPEDSIGNATURES)

Return the elements of a particular hyperparameter's history (identified by `nid`) that
correspond to trials whose losses were above or below the threshold.

Continuous parameters return `ObsPair{Observations}`; categorical / index parameters
return `ObsPair{IndexObjects.IndexVector}`. The pair type prevents swapping the good
(below) and bad (above) buckets.
"""
function ap_filter_trials(
    nid::Symbol, trials::Vector{Trials.Trial}, config::Config, ::Type{Int},
)::ObsPair{IndexObjects.IndexVector}
    below, above = _collect_ap_filter_trials(nid, trials, config)
    return ObsPair(
        IndexObjects.IndexVector(below),
        IndexObjects.IndexVector(above),
    )
end
function ap_filter_trials(
    nid::Symbol, trials::Vector{Trials.Trial}, config::Config, ::Type{Float64},
)::ObsPair{Observations}
    below, above = _collect_ap_filter_trials(nid, trials, config)
    return ObsPair(
        Observations(convert(Vector{Float64}, below)),
        Observations(convert(Vector{Float64}, above)),
    )
end

end # module ApFilterTrials
