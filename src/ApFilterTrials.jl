module ApFilterTrials

using DocStringExtensions

using ..Configuration
import ..Trials

"""
$(TYPEDSIGNATURES)

Return the elements of a particular hyperparameter's history (identified by `nid`) that
correspond to trials whose losses were above or below the threshold.
"""
function ap_filter_trials(
    nid::Symbol, trials::Vector{Trials.Trial}, config::Config
)::NTuple{2, Vector{T} where T <: Union{Float64, Int}}

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
    if all(isinteger.(below)) && all(isinteger.(above))
        below = convert(Vector{Int}, below)
        above = convert(Vector{Int}, above)
    else
        below = convert(Vector{Float64}, below)
        above = convert(Vector{Float64}, above)
    end
    return below, above
end

end # module ApFilterTrials
