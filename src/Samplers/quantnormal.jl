function quantnormal(
    obs::Observations, mu::Float64, sigma::Float64, q::Float64, sample_size::Int,
    config::Config
)::Tuple{PosteriorDraws, GMM.DistDetails}

    mixture = adaptive_parzen_normal(obs, mu, sigma, config)
    post = GMM.GMM1(mixture, q, sample_size)

    return PosteriorDraws(post), mixture
end
