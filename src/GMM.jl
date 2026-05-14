"""
Gaussian Mixture Models
"""
module GMM

import Distributions
using DocStringExtensions
import SpecialFunctions

function validate_mixture_args(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64}
)
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end
    Distributions.Categorical(weights) # validates weights (sum ≈ 1, non-negative)
    return nothing
end

function normal_cdf(
    x::Vector{Float64}, mu::Vector{Float64}, sigma::Vector{Float64}
)::Vector{Float64}

    if length(x) != length(mu) && length(x) != 1 && length(mu) != 1
        throw(DimensionMismatch("x: $(x) and mu: $(mu) are not the same length or 1"))
    end
    top = x .- mu
    bottom = max.(sqrt(2) .* sigma, eps(Float64))
    z = top ./ bottom

    return (1 .+ SpecialFunctions.erf.(z)) ./ 2
end
"""
$(TYPEDSIGNATURES)

Bounded Gaussian Mixture Model (BGMM)
Sample from truncated 1-D Gaussian Mixture Model

GMM1 with low, high
"""
function GMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    low::Float64, high::Float64, sample_size::Int
)::Vector{Float64}
    if low > high
        throw(ArgumentError(string(
            "low (", low, ") should not be greater than high ", high
        )))
    end
    validate_mixture_args(weights, mus, sigmas)
    if low >= high
        throw(ArgumentError("low is greater or equal to high, low: $(low), high: $(high)"))
    end
    d = Distributions.truncated(
        Distributions.MixtureModel(Distributions.Normal.(mus, sigmas), weights),
        low, high,
    )
    return rand(d, sample_size)
end
"""
$(TYPEDSIGNATURES)
GMM1 without low, high or q
"""
function GMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    sample_size::Int
)::Vector{Float64}
    validate_mixture_args(weights, mus, sigmas)
    d = Distributions.MixtureModel(Distributions.Normal.(mus, sigmas), weights)
    return rand(d, sample_size)
end
"""
$(TYPEDSIGNATURES)
GMM1 with low, high and q
"""
function GMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    low::Float64, high::Float64, q::Float64, sample_size::Int
)::Vector{Float64}
    samples = GMM1(weights, mus, sigmas, low, high, sample_size)

    return round.(samples ./ q) .* q
end
"""
$(TYPEDSIGNATURES)
GMM1 with q
"""
function GMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    q::Float64, sample_size::Int
)::Vector{Float64}
    samples = GMM1(weights, mus, sigmas, sample_size)

    return round.(samples ./ q) .* q
end

function logprob(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, low::Float64, high::Float64, q::Float64, p_accept::Float64
)::Vector{Float64}
    d = Distributions.MixtureModel(Distributions.Normal.(mus, sigmas), weights)
    ubound = min.(samples .+ (q / 2.0), high)
    lbound = max.(samples .- (q / 2.0), low)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end
function logprob(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, q::Float64, p_accept::Float64
)::Vector{Float64}
    d = Distributions.MixtureModel(Distributions.Normal.(mus, sigmas), weights)
    ubound = samples .+ (q / 2.0)
    lbound = samples .- (q / 2.0)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end

function logsum_rows(x::Matrix{Float64})::Vector{Float64}
    m = maximum(x, dims = 2)
    return (log.(sum(exp.(x .- m); dims = 2)) .+ m)[:]
end

function logpdf_mixture(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, p_accept::Float64
)::Vector{Float64}
    d = Distributions.MixtureModel(Distributions.Normal.(mus, sigmas), weights)
    return Distributions.logpdf.(Ref(d), samples) .- log(p_accept)
end

"""
$(TYPEDSIGNATURES)
GMM1_lpdf with low, high and q
"""
function GMM1_lpdf(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, low::Float64, high::Float64, q::Float64
)::Vector{Float64}
    isempty(samples) && return []
    validate_mixture_args(weights, mus, sigmas)
    p_accept = sum(weights .* (normal_cdf([high], mus, sigmas) - normal_cdf([low], mus, sigmas)))
    return logprob(samples, weights, mus, sigmas, low, high, q, p_accept)
end
"""
$(TYPEDSIGNATURES)
GMM1_lpdf with q
"""
function GMM1_lpdf(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, q::Float64
)::Vector{Float64}
    isempty(samples) && return []
    validate_mixture_args(weights, mus, sigmas)
    return logprob(samples, weights, mus, sigmas, q, 1.0)
end
"""
$(TYPEDSIGNATURES)
GMM1_lpdf with low, high
"""
function GMM1_lpdf(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, low::Float64, high::Float64
)::Vector{Float64}
    isempty(samples) && return []
    validate_mixture_args(weights, mus, sigmas)
    p_accept = sum(
        weights .* (normal_cdf([high], mus, sigmas) - normal_cdf([low], mus, sigmas))
    )
    return logpdf_mixture(samples, weights, mus, sigmas, p_accept)
end
"""
$(TYPEDSIGNATURES)
GMM1_lpdf without low, high or q
"""
function GMM1_lpdf(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}
)::Vector{Float64}
    isempty(samples) && return []
    validate_mixture_args(weights, mus, sigmas)
    return logpdf_mixture(samples, weights, mus, sigmas, 1.0)
end

end # module GMM
