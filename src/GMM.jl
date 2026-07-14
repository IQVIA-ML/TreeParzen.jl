"""
Gaussian Mixture Models
"""
module GMM

import Distributions
using DocStringExtensions
import SpecialFunctions

export DistDetails

"""
    DistDetails

Parameters of a 1-D Gaussian mixture: parallel vectors `weights`, `mus`, and `sigmas`.

Length and weight validity are checked at construction (`weights` must be `Float64` or
`Distributions.Categorical` / `MixtureModel` sampling will fail).
"""
struct DistDetails
    weights::Vector{Float64}
    mus::Vector{Float64}
    sigmas::Vector{Float64}

    function DistDetails(
        weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    )
        if !(length(weights) == length(mus) == length(sigmas))
            throw(DimensionMismatch(string(
                "length(weights): ", length(weights),
                " doesn't equal length(mus): ", length(mus),
                " nor length(sigmas): ", length(sigmas),
            )))
        end
        Distributions.Categorical(weights) # validates weights (sum ≈ 1, non-negative)
        return new(weights, mus, sigmas)
    end
end

function normal_cdf(x::Vector{Float64}, mixture::DistDetails)::Vector{Float64}
    m = mixture.mus
    s = mixture.sigmas
    if length(x) != length(m) && length(x) != 1 && length(m) != 1
        throw(DimensionMismatch("x: $(x) and mu: $(m) are not the same length or 1"))
    end
    top = x .- m
    bottom = max.(sqrt(2) .* s, eps(Float64))
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
    mixture::DistDetails, low::Float64, high::Float64, sample_size::Int
)::Vector{Float64}
    if low > high
        throw(ArgumentError(string(
            "low (", low, ") should not be greater than high ", high
        )))
    end
    if low >= high
        throw(ArgumentError("low is greater or equal to high, low: $(low), high: $(high)"))
    end
    d = Distributions.truncated(
        Distributions.MixtureModel(
            Distributions.Normal.(mixture.mus, mixture.sigmas), mixture.weights
        ),
        low, high,
    )
    return rand(d, sample_size)
end
"""
$(TYPEDSIGNATURES)
GMM1 without low, high or q
"""
function GMM1(mixture::DistDetails, sample_size::Int)::Vector{Float64}
    d = Distributions.MixtureModel(
        Distributions.Normal.(mixture.mus, mixture.sigmas), mixture.weights
    )
    return rand(d, sample_size)
end
"""
$(TYPEDSIGNATURES)
GMM1 with low, high and q
"""
function GMM1(
    mixture::DistDetails, low::Float64, high::Float64, q::Float64, sample_size::Int
)::Vector{Float64}
    samples = GMM1(mixture, low, high, sample_size)

    return round.(samples ./ q) .* q
end
"""
$(TYPEDSIGNATURES)
GMM1 with q
"""
function GMM1(mixture::DistDetails, q::Float64, sample_size::Int)::Vector{Float64}
    samples = GMM1(mixture, sample_size)

    return round.(samples ./ q) .* q
end

function logprob(
    samples::Vector{Float64}, mixture::DistDetails, low::Float64, high::Float64, q::Float64,
    p_accept::Float64
)::Vector{Float64}
    d = Distributions.MixtureModel(
        Distributions.Normal.(mixture.mus, mixture.sigmas), mixture.weights
    )
    ubound = min.(samples .+ (q / 2.0), high)
    lbound = max.(samples .- (q / 2.0), low)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end
function logprob(
    samples::Vector{Float64}, mixture::DistDetails, q::Float64, p_accept::Float64
)::Vector{Float64}
    d = Distributions.MixtureModel(
        Distributions.Normal.(mixture.mus, mixture.sigmas), mixture.weights
    )
    ubound = samples .+ (q / 2.0)
    lbound = samples .- (q / 2.0)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end

function logsum_rows(x::Matrix{Float64})::Vector{Float64}
    m = maximum(x, dims = 2)
    return (log.(sum(exp.(x .- m); dims = 2)) .+ m)[:]
end

# Vectorized log pdf of a 1D Gaussian mixture at `samples` (faster than logpdf(MixtureModel, ...) per point).
function mahal(
    samples::Vector{Float64}, mixture::DistDetails, p_accept::Float64
)::Vector{Float64}
    m = mixture.mus
    s = mixture.sigmas
    w = mixture.weights
    dist = reshape(samples, length(samples), 1) .- reshape(m, 1, length(m))
    # mahal size is (n_samples, n_components)
    mahal = (dist ./ reshape(max.(s, eps(Float64)), 1, length(s))) .^ 2
    Z = sqrt.(2pi .* (s .^ 2))
    coef = w ./ Z ./ p_accept

    return logsum_rows(-0.5 .* mahal .+ log.(reshape(coef, 1, length(coef))))
end

"""
$(TYPEDSIGNATURES)
GMM1_lpdf with low, high and q
"""
function GMM1_lpdf(
    samples::Vector{Float64}, mixture::DistDetails, low::Float64, high::Float64, q::Float64
)::Vector{Float64}
    isempty(samples) && return []
    p_accept = sum(
        mixture.weights .* (normal_cdf([high], mixture) - normal_cdf([low], mixture))
    )
    return logprob(samples, mixture, low, high, q, p_accept)
end
"""
$(TYPEDSIGNATURES)
GMM1_lpdf with q
"""
function GMM1_lpdf(
    samples::Vector{Float64}, mixture::DistDetails, q::Float64
)::Vector{Float64}
    isempty(samples) && return []
    p_accept = 1.0

    return logprob(samples, mixture, q, p_accept)
end
"""
$(TYPEDSIGNATURES)
GMM1_lpdf with low, high
"""
function GMM1_lpdf(
    samples::Vector{Float64}, mixture::DistDetails, low::Float64, high::Float64
)::Vector{Float64}
    isempty(samples) && return []
    p_accept = sum(
        mixture.weights .* (normal_cdf([high], mixture) - normal_cdf([low], mixture))
    )

    return mahal(samples, mixture, p_accept)
end
"""
$(TYPEDSIGNATURES)
GMM1_lpdf without low, high or q
"""
function GMM1_lpdf(samples::Vector{Float64}, mixture::DistDetails)::Vector{Float64}
    isempty(samples) && return []

    return mahal(samples, mixture, 1.0)
end

end # module GMM
