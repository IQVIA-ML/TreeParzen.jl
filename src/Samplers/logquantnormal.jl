function logquantnormal(
    obs::Observations, mu::Float64, sigma::Float64, q::Float64, sample_size::Int,
    config::Config
)::Tuple{LogPosteriorDraws, GMM.DistDetails}

    mixture = adaptive_parzen_normal(
        Observations(log.(max.(obs.v, eps(Float64)))), mu, sigma, config
    )
    post = LogGMM.LGMM1(mixture, q, sample_size)

    return LogPosteriorDraws(post), mixture
end

logquantnormal(
    obs::AbstractVector{<:Real}, mu::Float64, sigma::Float64, q::Float64, sample_size::Int,
    config::Config,
) = logquantnormal(Observations(obs), mu, sigma, q, sample_size, config)
