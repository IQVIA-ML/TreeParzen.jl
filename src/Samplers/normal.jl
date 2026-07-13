function normal(
    obs::Vector{Float64}, mu::Float64, sigma::Float64, sample_size::Int, config::Config
)::Tuple{Vector{Float64}, GMM.DistDetails}

    mixture = adaptive_parzen_normal(obs, mu, sigma, config)
    post = GMM.GMM1(mixture, sample_size)

    return post, mixture
end
