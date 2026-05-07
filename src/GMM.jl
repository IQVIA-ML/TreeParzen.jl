"""
Gaussian Mixture Models
"""
module GMM

import Distributions
using DocStringExtensions
import SpecialFunctions

function mixture_model(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64}
)::Distributions.MixtureModel
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end
    return Distributions.MixtureModel(Distributions.Normal.(mus, sigmas), weights)
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
    # validate (including weights)
    mixture_model(weights, mus, sigmas) 
    if low >= high
        throw(ArgumentError("low is greater or equal to high, low: $(low), high: $(high)"))
    end
    d = Distributions.truncated(mixture_model(weights, mus, sigmas), low, high)
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
    d = mixture_model(weights, mus, sigmas)
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
    d = mixture_model(weights, mus, sigmas)
    ubound = min.(samples .+ (q / 2.0), high)
    lbound = max.(samples .- (q / 2.0), low)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end
function logprob(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, q::Float64, p_accept::Float64
)::Vector{Float64}
    d = mixture_model(weights, mus, sigmas)
    ubound = samples .+ (q / 2.0)
    lbound = samples .- (q / 2.0)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end

function logsum_rows(x::Matrix{Float64})::Vector{Float64}
    m = maximum(x, dims = 2)

    return (log.(sum(exp.(x .- m); dims = 2)) .+ m)[:]
end

function mahal(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, p_accept::Float64
)::Vector{Float64}
    d = mixture_model(weights, mus, sigmas)
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
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end

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
    mixture_model(weights, mus, sigmas) # validate (including weights)
    p_accept = 1.0

    return logprob(samples, weights, mus, sigmas, q, p_accept)
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
    mixture_model(weights, mus, sigmas) # validate (including weights)
    p_accept = sum(
        weights .* (normal_cdf([high], mus, sigmas) - normal_cdf([low], mus, sigmas))
    )

    return mahal(samples, weights, mus, sigmas, p_accept)
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
    mixture_model(weights, mus, sigmas) # validate (including weights)
    return mahal(samples, weights, mus, sigmas, 1.0)
end

end # module GMM
