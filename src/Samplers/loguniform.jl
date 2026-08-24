function loguniform(
    obs::Observations, low::Float64, high::Float64, sample_size::Int, config::Config
)::Tuple{LogPosteriorDraws, GMM.DistDetails}

    prior_mu = (high + low) / 2
    prior_sigma = high - low
    mixture = adaptive_parzen_normal(
        Observations(log.(obs.v)), prior_mu, prior_sigma, config
    )
    post = LogGMM.LGMM1(mixture, low, high, sample_size)

    return LogPosteriorDraws(post), mixture
end

loguniform(
    obs::AbstractVector{<:Real}, low::Float64, high::Float64, sample_size::Int, config::Config,
) = loguniform(Observations(obs), low, high, sample_size, config)
