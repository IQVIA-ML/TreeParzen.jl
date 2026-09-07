function uniform(
    obs::Observations, low::Float64, high::Float64, sample_size::Int, config::Config
)::Tuple{PosteriorDraws, GMM.DistDetails}

    prior_mu = 1//2 * (high + low)
    prior_sigma = high - low
    mixture = adaptive_parzen_normal(obs, prior_mu, prior_sigma, config)
    post = GMM.GMM1(mixture, low, high, sample_size)

    return PosteriorDraws(post), mixture
end
