function categorical_pseudocounts(
    counts::Vector{Float64}, prior_weight::Real, probabilities::Probabilities,
    sample_size::Int
)::Probabilities
    if iszero(sample_size)
        return Probabilities(Float64[])
    end

    if length(counts) != length(probabilities)
        throw(DimensionMismatch("counts and probs are different lengths"))
    end
    # Blend empirical counts with a prior; normalise to a simplex.
    blended = counts + length(probabilities) * (prior_weight * probabilities.v)

    return Probabilities(blended / sum(blended))
end

function categoricalindex(
    obs::IndexObjects.IndexVector, probabilities::Probabilities, sample_size::Int, config::Config
)::Tuple{IndexObjects.IndexVector, Probabilities}

    weights = ForgettingWeights.forgetting_weights(
        length(obs.v), config.linear_forgetting
    )
    counts = Bincounts.bincount(obs.v, weights, length(probabilities))
    posterior_probs = categorical_pseudocounts(
        counts, config.prior_weight, probabilities, sample_size
    )
    post = Delayed.categoricalindex(posterior_probs, sample_size)

    return post, posterior_probs
end
