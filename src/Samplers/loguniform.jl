function loguniform(
    obs::Vector{Float64}, low::Float64, high::Float64,sample_size::Int, config::Config
)::Tuple{Matrix{Float64}, GMM.DistDetails}

    prior_mu = (high + low) / 2
    prior_sigma = high - low
    mixture = adaptive_parzen_normal(
        log.(obs), prior_mu, prior_sigma, config
    )
    post = LogGMM.LGMM1(mixture, low, high, sample_size)

    return post, mixture

end
