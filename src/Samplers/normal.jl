function normal(
    obs::Observations, mu::Float64, sigma::Float64, sample_size::Int, config::Config
)::Tuple{PosteriorDraws, GMM.DistDetails}

    mixture = adaptive_parzen_normal(obs, mu, sigma, config)
    post = GMM.GMM1(mixture, sample_size)

    return PosteriorDraws(post), mixture
end
