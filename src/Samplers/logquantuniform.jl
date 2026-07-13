function logquantuniform(
    obs::Vector{Float64}, low::Float64, high::Float64, q::Float64, sample_size::Int,
    config::Config
)::Tuple{Matrix{Float64}, GMM.DistDetails}

    prior_mu = (high + low) / 2
    prior_sigma = high - low
    components = adaptive_parzen_normal(log.(max.(obs, eps(Float64))),
     prior_mu, prior_sigma, config)
    post = LogGMM.LGMM1(components, low, high, q, sample_size)

    return post, components
end
