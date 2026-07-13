function uniform(
    obs::Vector{Float64}, low::Float64, high::Float64, sample_size::Int, config::Config
)::Tuple{Vector{Float64}, GMM.DistDetails}

    prior_mu = 1//2 * (high + low)
    prior_sigma = high - low
    components = adaptive_parzen_normal(obs, prior_mu, prior_sigma, config)
    post = GMM.GMM1(components, low, high, sample_size)

    return post, components
end
