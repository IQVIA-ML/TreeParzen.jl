function randint(
    obs::Vector{Int}, upper::Int, sample_size::Int, config::Config
)::Tuple{Vector{Int}, Vector{Float64}}

    weights = ForgettingWeights.forgetting_weights(
        length(obs), config.linear_forgetting
    )
    counts = Bincounts.bincount(obs .+ 1, weights, upper)
    # -- add in some prior pseudocounts
    pseudocounts = counts .+ config.prior_weight
    probabilities = pseudocounts / sum(pseudocounts)
    post = Delayed.categoricalindex(probabilities, sample_size).v .- 1

    return post, probabilities
end
