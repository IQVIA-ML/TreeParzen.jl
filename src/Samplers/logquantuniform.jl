function logquantuniform(
    obs::Observations, low::Float64, high::Float64, q::Float64, sample_size::Int,
    config::Config
)::Tuple{LogPosteriorDraws, GMM.DistDetails}

    prior_mu = (high + low) / 2
    prior_sigma = high - low
    mixture = adaptive_parzen_normal(
        Observations(log.(max.(obs.v, eps(Float64)))), prior_mu, prior_sigma, config
    )
    post = LogGMM.LGMM1(mixture, low, high, q, sample_size)

    return LogPosteriorDraws(post), mixture
end

logquantuniform(
    obs::AbstractVector{<:Real}, low::Float64, high::Float64, q::Float64, sample_size::Int,
    config::Config,
) = logquantuniform(Observations(obs), low, high, q, sample_size, config)
