function lognormal(
    obs::Vector{Float64}, mu::Float64, sigma::Float64, sample_size::Int, config::Config
)::Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

    weights, mus, sigmas = adaptive_parzen_normal(log.(obs), mu, sigma, config)
    post = LogGMM.LGMM1(weights, mus, sigmas, sample_size)

    return post, weights, mus, sigmas
end
