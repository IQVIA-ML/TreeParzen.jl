function normal(
    obs::Vector{Float64}, mu::Float64, sigma::Float64, sample_size::Int, config::Config
)::Tuple{Vector{Float64}, Vector{GMM.DistDetails}}

    components = adaptive_parzen_normal(obs, mu, sigma, config)
    post = GMM.GMM1(components, sample_size)

    return post, components
end
