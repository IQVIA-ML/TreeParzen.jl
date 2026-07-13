function lognormal(
    obs::Vector{Float64}, mu::Float64, sigma::Float64, sample_size::Int, config::Config
)::Tuple{Matrix{Float64}, GMM.DistDetails}

    components = adaptive_parzen_normal(log.(obs), mu, sigma, config)
    post = LogGMM.LGMM1(components, sample_size)

    return post, components
end
