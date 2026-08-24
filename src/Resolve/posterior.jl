function categorical_lpdf(
    sample::IndexObjects.IndexVector, probabilities::Probabilities
)::Vector{Float64}
    isempty(sample.v) && return Float64[]

    if maximum(sample.v) > length(probabilities)
        throw(DimensionMismatch(string(
            "maximum sample value (", maximum(sample.v),
            ") larger than length of probabilities (", length(probabilities), "), but ",
            "will be used to index. Values in sample: ", unique(sample.v)
        )))
    end

    return [log(probabilities[x]) for x in sample.v]
end

categorical_lpdf(sample::IndexObjects.IndexVector, probabilities::AbstractVector{<:Real}) =
    categorical_lpdf(sample, Probabilities(probabilities))

function posterior(
    node::Delayed.CategoricalIndex, probabilities::Vector{Float64}, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::IndexObjects.IndexInt
    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Int)
    prior = Probabilities(probabilities)

    b_post, b_probs = Samplers.categoricalindex(
        obs.below, prior, config.draws, config
    )
    _, a_probs = Samplers.categoricalindex(
        obs.above, prior, config.draws, config
    )

    if isempty(b_post.v)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = categorical_lpdf(b_post, b_probs)
    above_llik = categorical_lpdf(b_post, a_probs)

    return IndexObjects.IndexInt(b_post.v[argmax(below_llik .- above_llik)])
end
function posterior(
    node::Delayed.LogNormal, mu::Float64, sigma::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Float64)

    b_post, b_mixture = Samplers.lognormal(
        obs.below, mu, sigma, config.draws, config
    )
    _, a_mixture = Samplers.lognormal(
        obs.above, mu, sigma, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = LogGMM.LGMM1_lpdf(b_post.v, b_mixture)
    above_llik = LogGMM.LGMM1_lpdf(b_post.v, a_mixture)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.LogNormalQuantDist, mu::Float64, sigma::Float64, q::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Float64)

    b_post, b_mixture = Samplers.logquantnormal(
        obs.below, mu, sigma, q, config.draws, config
    )
    _, a_mixture = Samplers.logquantnormal(
        obs.above, mu, sigma, q, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = LogGMM.LGMM1_lpdf(b_post.v, b_mixture, q)
    above_llik = LogGMM.LGMM1_lpdf(b_post.v, a_mixture, q)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.Normal, mu::Float64, sigma::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Float64)

    b_post, b_mixture = Samplers.normal(
        obs.below, mu, sigma, config.draws, config
    )
    _, a_mixture = Samplers.normal(
        obs.above, mu, sigma, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = GMM.GMM1_lpdf(b_post.v, b_mixture)
    above_llik = GMM.GMM1_lpdf(b_post.v, a_mixture)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.QuantNormal, mu::Float64, sigma::Float64, q::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Float64)

    b_post, b_mixture = Samplers.quantnormal(
        obs.below, mu, sigma, q, config.draws, config
    )
    _, a_mixture = Samplers.quantnormal(
        obs.above, mu, sigma, q, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = GMM.GMM1_lpdf(b_post.v, b_mixture, q)
    above_llik = GMM.GMM1_lpdf(b_post.v, a_mixture, q)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.RandIndex, upper::Int, nid::Symbol, trials::Vector{Trials.Trial},
    config::Config
)::IndexObjects.IndexInt

    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Int)

    b_post, b_probabilities = Samplers.randindex(
        obs.below, upper, config.draws, config
    )
    _, a_probabilities = Samplers.randindex(
        obs.above, upper, config.draws, config
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

    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Float64)

    b_post, b_mixture = Samplers.uniform(
        obs.below, low, high, config.draws, config
    )
    _, a_mixture = Samplers.uniform(
        obs.above, low, high, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = GMM.GMM1_lpdf(b_post.v, b_mixture, low, high)
    above_llik = GMM.GMM1_lpdf(b_post.v, a_mixture, low, high)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.QuantUniform, low::Float64, high::Float64, q::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Float64)

    b_post, b_mixture = Samplers.quantuniform(
        obs.below, low, high, q, config.draws, config
    )
    _, a_mixture = Samplers.quantuniform(
        obs.above, low, high, q, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = GMM.GMM1_lpdf(b_post.v, b_mixture, low, high, q)
    above_llik = GMM.GMM1_lpdf(b_post.v, a_mixture, low, high, q)

    return b_post[argmax(below_llik .- above_llik)]
end

function posterior(
    node::Delayed.LogUniform, low::Float64, high::Float64, nid::Symbol, trials::Vector{Trials.Trial},
    config::Config
)::Real

    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Float64)

    b_post, b_mixture = Samplers.loguniform(
        obs.below, low, high, config.draws, config
    )
    _, a_mixture = Samplers.loguniform(
        obs.above, low, high, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = LogGMM.LGMM1_lpdf(b_post.v, b_mixture)
    above_llik = LogGMM.LGMM1_lpdf(b_post.v, a_mixture)

    return b_post[argmax(below_llik .- above_llik)]
end
function posterior(
    node::Delayed.LogUniformQuantDist, low::Float64, high::Float64, q::Float64, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    obs = ApFilterTrials.ap_filter_trials(nid, trials, config, Float64)

    b_post, b_mixture = Samplers.logquantuniform(
        obs.below, low, high, q, config.draws, config
    )
    _, a_mixture = Samplers.logquantuniform(
        obs.above, low, high, q, config.draws, config
    )

    if isempty(b_post)
        throw(ArgumentError("b_post is empty"))
    end

    below_llik = LogGMM.LGMM1_lpdf(b_post.v, b_mixture, low, high, q)
    above_llik = LogGMM.LGMM1_lpdf(b_post.v, a_mixture, low, high, q)

    return b_post[argmax(below_llik .- above_llik)]
end
