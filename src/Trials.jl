module Trials

using DocStringExtensions

const ValsDict = Dict{Symbol, Real}

"""
$(TYPEDEF)

Contains the suggested hyperparameters and the resulting loss for a particular trial.

$(TYPEDFIELDS)

# Arguments

- `hyperparams` : suggested hyperparameters for the trial stored in a form that can be
    submitted to the user's function
- `vals` : suggested hyperparameters for the trial stored in a form that can be learnt from.
    e.g. choices are indexs rather than choice values
- `loss` : the float returned by evaluating the user's function
"""
mutable struct Trial
    hyperparams::Dict{Symbol, T} where T
    vals::ValsDict
    loss::Float64
end

end # module Trials
