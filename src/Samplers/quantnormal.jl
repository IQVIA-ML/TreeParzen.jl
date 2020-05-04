function quantnormal(
    obs::Vector{Float64}, mu::Float64, sigma::Float64, q::Float64, sample_size::Int,
    config::Config
)::NTuple{4, Vector{Float64}}

    weights, mus, sigmas = adaptive_parzen_normal(obs, mu, sigma, config)
    post = GMM.GMM1(weights, mus, sigmas, q, sample_size)

    return post, weights, mus, sigmas
end
