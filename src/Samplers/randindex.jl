function randindex(
    obs::IndexObjects.IndexVector, upper::Int, sample_size::Int, config::Config
)::Tuple{IndexObjects.IndexVector, Probabilities}

    weights = ForgettingWeights.forgetting_weights(
        length(obs.v), config.linear_forgetting
    )
    counts = Bincounts.bincount(obs.v, weights, upper)
    # -- add in some prior pseudocounts
    blended = counts .+ config.prior_weight
    probabilities = Probabilities(blended / sum(blended))
    post = Delayed.categoricalindex(probabilities, sample_size)
    return post, probabilities
end
