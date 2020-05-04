function uniform(
    obs::Vector{Float64}, low::Float64, high::Float64, sample_size::Int, config::Config
)::NTuple{4, Vector{Float64}}

    prior_mu = 1//2 * (high + low)
    prior_sigma = high - low
    weights, mus, sigmas = adaptive_parzen_normal(obs, prior_mu, prior_sigma, config)
    post = GMM.GMM1(weights, mus, sigmas, low, high, sample_size)

    return post, weights, mus, sigmas
end
