function quantuniform(
    obs::Vector{Float64}, low::Float64, high::Float64, q::Float64, sample_size::Int,
    config::Config
)::Tuple{Vector{Float64}, GMM.DistDetails}

    prior_mu = (high + low) / 2
    prior_sigma = high - low
    mixture = adaptive_parzen_normal(obs, prior_mu, prior_sigma, config)
    post = GMM.GMM1(mixture, low, high, q, sample_size)

    return post, mixture
end
