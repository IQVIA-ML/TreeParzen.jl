function loguniform(
    obs::Vector{Float64}, low::Float64, high::Float64,sample_size::Int, config::Config
)::Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

    prior_mu = (high + low) / 2
    prior_sigma = high - low
    weights, mus, sigmas = adaptive_parzen_normal(
        log.(obs), prior_mu, prior_sigma, config
    )
    post = LogGMM.LGMM1(weights, mus, sigmas, low, high, sample_size)

    return post, weights, mus, sigmas

end
