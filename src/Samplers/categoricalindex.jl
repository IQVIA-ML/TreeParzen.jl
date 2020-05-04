function categorical_pseudocounts(
    counts::Vector{Float64}, prior_weight::Real, probabilities::Vector{Float64},
    sample_size::Int
)::Vector{Float64}
    if iszero(sample_size)
        return []
    end

    if length(counts) != length(probabilities)
        throw(DimensionMismatch("counts and probs are different lengths"))
    end
    pseudocounts = counts + length(probabilities) * (prior_weight * probabilities)

    return pseudocounts / sum(pseudocounts)
end

function categoricalindex(
    obs::IndexObjects.IndexVector, probabilities::Vector{Float64}, sample_size::Int, config::Config
)::Tuple{IndexObjects.IndexVector, Vector{Float64}}

    weights = LinearForgettingWeights.linear_forgetting_weights(
        length(obs.v), config.linear_forgetting
    )
    counts = Bincounts.bincount(obs.v, weights, length(probabilities))
    pseudocounts = categorical_pseudocounts(
        counts, config.prior_weight, probabilities, sample_size
    )
    post = Delayed.categoricalindex(pseudocounts, sample_size)

    return post, pseudocounts
end
