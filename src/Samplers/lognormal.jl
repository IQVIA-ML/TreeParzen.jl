function lognormal(
    obs::Observations, mu::Float64, sigma::Float64, sample_size::Int, config::Config
)::Tuple{LogPosteriorDraws, GMM.DistDetails}

    mixture = adaptive_parzen_normal(Observations(log.(obs.v)), mu, sigma, config)
    post = LogGMM.LGMM1(mixture, sample_size)

    return LogPosteriorDraws(post), mixture
end

lognormal(obs::AbstractVector{<:Real}, mu::Float64, sigma::Float64, sample_size::Int, config::Config) =
    lognormal(Observations(obs), mu, sigma, sample_size, config)
