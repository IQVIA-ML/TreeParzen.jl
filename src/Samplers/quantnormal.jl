function quantnormal(
    obs::Vector{Float64}, mu::Float64, sigma::Float64, q::Float64, sample_size::Int,
    config::Config
)::Tuple{Vector{Float64}, GMM.DistDetails}

    components = adaptive_parzen_normal(obs, mu, sigma, config)
    post = GMM.GMM1(components, q, sample_size)

    return post, components
end
