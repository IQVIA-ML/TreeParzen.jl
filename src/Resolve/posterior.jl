function categorical_lpdf(
    sample::IndexObjects.IndexVector, probabilities::Vector{Float64}
)::Vector{Float64}
    isempty(sample.v) && return Float64[]

    if maximum(sample.v) > length(probabilities)
        throw(DimensionMismatch(string(
            "maximum sample value (", maximum(sample),
            ") larger than length of probabilities (", length(probabilities), "), but ",
            "will be used to index. Values in sample: ", unique(sample)
        )))
    end

    return [log(probabilities[x]) for x in sample.v]
end

function posterior(
    node::Delayed.CategoricalIndex, probabilities::Vector{Float64}, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::IndexObjects.IndexInt
    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_pseudocounts = Samplers.categoricalindex(
        IndexObjects.IndexVector(obs_below), probabilities, config.draws, config
    )
    _, a_pseudocounts = Samplers.categoricalindex(
        IndexObjects.IndexVector(obs_above), probabilities, config.draws, config
    )

    if isempty(b_post.v)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = categorical_lpdf(b_post, b_pseudocounts)
    above_llik = categorical_lpdf(b_post, a_pseudocounts)

    return IndexObjects.IndexInt(b_post.v[argmax(below_llik .- above_llik)])
end
function posterior(
    node::Delayed.LogNormal, mu::Float64, sigma::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_weights, b_mus, b_sigmas = Samplers.lognormal(
        obs_below, mu, sigma, config.draws, config
    )
    _, a_weights, a_mus, a_sigmas = Samplers.lognormal(
        obs_above, mu, sigma, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = LogGMM.LGMM1_lpdf(b_post, b_weights, b_mus, b_sigmas)
    above_llik = LogGMM.LGMM1_lpdf(b_post, a_weights, a_mus, a_sigmas)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.LogQuantNormal, mu::Float64, sigma::Float64, q::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_weights, b_mus, b_sigmas = Samplers.logquantnormal(
        Float64.(obs_below), mu, sigma, q, config.draws, config
    )
    _, a_weights, a_mus, a_sigmas = Samplers.logquantnormal(
        Float64.(obs_above), mu, sigma, q, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = LogGMM.LGMM1_lpdf(b_post, b_weights, b_mus, b_sigmas, q)
    above_llik = LogGMM.LGMM1_lpdf(b_post, a_weights, a_mus, a_sigmas, q)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.Normal, mu::Float64, sigma::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_weights, b_mus, b_sigmas = Samplers.normal(
        Float64.(obs_below), mu, sigma, config.draws, config
    )
    _, a_weights, a_mus, a_sigmas = Samplers.normal(
        Float64.(obs_above), mu, sigma, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = GMM.GMM1_lpdf(b_post, b_weights, b_mus, b_sigmas)
    above_llik = GMM.GMM1_lpdf(b_post, a_weights, a_mus, a_sigmas)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.QuantNormal, mu::Float64, sigma::Float64, q::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_weights, b_mus, b_sigmas = Samplers.quantnormal(
        Float64.(obs_below), mu, sigma, q, config.draws, config
    )
    _, a_weights, a_mus, a_sigmas = Samplers.quantnormal(
        Float64.(obs_above), mu, sigma, q, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = GMM.GMM1_lpdf(b_post, b_weights, b_mus, b_sigmas, q)
    above_llik = GMM.GMM1_lpdf(b_post, a_weights, a_mus, a_sigmas, q)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.RandIndex, upper::Int, nid::Symbol, trials::Vector{Trials.Trial},
    config::Config
)::IndexObjects.IndexInt

    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_probabilities = Samplers.randindex(
        IndexObjects.IndexVector(obs_below), upper, config.draws, config
    )
    _, a_probabilities = Samplers.randindex(
        IndexObjects.IndexVector(obs_above), upper, config.draws, config
    )
    if isempty(b_post.v)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = categorical_lpdf(b_post, b_probabilities)
    above_llik = categorical_lpdf(b_post, a_probabilities)

    return IndexObjects.IndexInt(b_post.v[argmax(below_llik .- above_llik)])
end
function posterior(
    node::Delayed.Uniform, low::Float64, high::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_weights, b_mus, b_sigmas = Samplers.uniform(
        obs_below, low, high, config.draws, config
    )
    _, a_weights, a_mus, a_sigmas = Samplers.uniform(
        obs_above, low, high, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = GMM.GMM1_lpdf(b_post, b_weights, b_mus, b_sigmas, low, high)
    above_llik = GMM.GMM1_lpdf(b_post, a_weights, a_mus, a_sigmas, low, high)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.QuantUniform, low::Float64, high::Float64, q::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_weights, b_mus, b_sigmas = Samplers.quantuniform(
        Float64.(obs_below), low, high, q, config.draws, config
    )
    _, a_weights, a_mus, a_sigmas = Samplers.quantuniform(
        Float64.(obs_above), low, high, q, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = GMM.GMM1_lpdf(b_post, b_weights, b_mus, b_sigmas, low, high, q)
    above_llik = GMM.GMM1_lpdf(b_post, a_weights, a_mus, a_sigmas, low, high, q)

    return b_post[argmax(below_llik .- above_llik)]
end

function posterior(
    node::Delayed.LogUniform, low::Float64, high::Float64, nid::Symbol, trials::Vector{Trials.Trial},
    config::Config
)::Real

    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_weights, b_mus, b_sigmas = Samplers.loguniform(
        float.(obs_below), low, high, config.draws, config
    )
    _, a_weights, a_mus, a_sigmas = Samplers.loguniform(
        float.(obs_above), low, high, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = LogGMM.LGMM1_lpdf(
        b_post, b_weights, b_mus, b_sigmas, low, high
    )
    above_llik = LogGMM.LGMM1_lpdf(
        b_post, a_weights, a_mus, a_sigmas, low, high
    )

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.LogQuantUniform, low::Float64, high::Float64, q::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs_below, obs_above = ApFilterTrials.ap_filter_trials(nid, trials, config)

    b_post, b_weights, b_mus, b_sigmas = Samplers.logquantuniform(
        float.(obs_below), low, high, q, config.draws, config
    )
    _, a_weights, a_mus, a_sigmas = Samplers.logquantuniform(
        float.(obs_above), low, high, q, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = LogGMM.LGMM1_lpdf(b_post, b_weights, b_mus, b_sigmas,low, high, q)
    above_llik = LogGMM.LGMM1_lpdf(b_post, a_weights, a_mus, a_sigmas,low, high, q)

    return b_post[argmax(below_llik .- above_llik)]
end
