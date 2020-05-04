function randindex(
    obs::IndexObjects.IndexVector, upper::Int, sample_size::Int, config::Config
)::Tuple{IndexObjects.IndexVector, Vector{Float64}}

    weights = LinearForgettingWeights.linear_forgetting_weights(
        length(obs.v), config.linear_forgetting
    )
    counts = Bincounts.bincount(obs.v, weights, upper)
    # -- add in some prior pseudocounts
    pseudocounts = counts .+ config.prior_weight
    probabilities = pseudocounts / sum(pseudocounts)
    post = Delayed.categoricalindex(probabilities, sample_size)
    return post, probabilities
end
