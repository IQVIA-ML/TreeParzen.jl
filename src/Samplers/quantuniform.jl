function quantuniform(
    obs::Observations, low::Float64, high::Float64, q::Float64, sample_size::Int,
    config::Config
)::Tuple{PosteriorDraws, GMM.DistDetails}

    prior_mu = (high + low) / 2
    prior_sigma = high - low
    mixture = adaptive_parzen_normal(obs, prior_mu, prior_sigma, config)
    post = GMM.GMM1(mixture, low, high, q, sample_size)

    return PosteriorDraws(post), mixture
end

quantuniform(
    obs::AbstractVector{<:Real}, low::Float64, high::Float64, q::Float64, sample_size::Int,
    config::Config,
) = quantuniform(Observations(obs), low, high, q, sample_size, config)
