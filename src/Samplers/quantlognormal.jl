function quantlognormal(
    obs::Vector{Float64}, mu::Float64, sigma::Float64, q::Float64, sample_size::Int,
    config::Config
)::Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

    weights, mus, sigmas = adaptive_parzen_normal(
        log.(max.(obs, eps(Float64))), mu, sigma, config
    )
    post = LogGMM.LGMM1(weights, mus, sigmas, q, sample_size)

    return post, weights, mus, sigmas
end
