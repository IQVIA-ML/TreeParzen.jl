module Bincounts

using DocStringExtensions
using Distributions

"""
$(TYPEDSIGNATURES)

Similar to numpy.bincount:
(https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html)

Count the weighted occurrences of each integer from one to the maximum integer in obs.
obs cannot contain integers below 1.

Specifying a minlength will guarantee at least this length of output - equivalent to
creating counts for integers above the maximum integer in obs.
Each location of the output gives the count of occurrences of its index value in obs,
i.e. it represents a number line starting from one.

Counts of occurences in obs are weighted by weights, therefore obs and weights must be the
same length. A weights consisting entirely of 1.0 is equivalent to calling without weights.
"""
function bincount(
    obs::Vector{Int}, weights::Vector{Float64}, minlength::Int
)::Vector{Float64}

    if isempty(obs)
        return ones(minlength)
    end

    if minimum(obs) < 1
        throw(DomainError("obs must be greater than 0"))
    end

    if length(obs) != length(weights)
        throw(DimensionMismatch(string(
            "The number of observations: ", length(obs),
            " is not equal to the number of weights: ", length(weights)
        )))
    end

    return Distributions.fit(
        Distributions.Histogram, obs, Distributions.Weights(weights),
        1:max(minlength + 1, maximum(obs) + 1)
    ).weights
end
bincount(obs::Vector{Int}, minlength::Int)::Vector{Float64} = bincount(obs, ones(size(obs)), minlength)

end # module Bincounts
