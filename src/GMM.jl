"""
Gaussian Mixture Models
"""
module GMM

import Distributions
using DocStringExtensions
import SpecialFunctions

export DistDetails, mixture

"""
    DistDetails

Parameters of one component in a 1-D Gaussian mixture: `weight`, `mu`, and `sigma`.
"""
struct DistDetails
    weight::Float64
    mu::Float64
    sigma::Float64
end

"""
    mixture(weights, mus, sigmas) -> Vector{DistDetails}

Build a mixture from parallel vectors of component parameters.

`weights` must be `Float64` or `Distributions.Categorical` / `MixtureModel` sampling will fail.
"""
function mixture(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
)::Vector{DistDetails}
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end
    return [DistDetails(weights[i], mus[i], sigmas[i]) for i in eachindex(weights)]
end

weights(c::Vector{DistDetails}) = [x.weight for x in c]
mus(c::Vector{DistDetails}) = [x.mu for x in c]
sigmas(c::Vector{DistDetails}) = [x.sigma for x in c]

function validate_mixture_args(components::Vector{DistDetails})
    # weights must be Float64 or Categorical / MixtureModel sampling will fail
    Distributions.Categorical(weights(components)) # validates weights (sum ≈ 1, non-negative)
    return nothing
end

function normal_cdf(
    x::Vector{Float64}, components::Vector{DistDetails}
)::Vector{Float64}
    m = mus(components)
    s = sigmas(components)
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
    components::Vector{DistDetails}, low::Float64, high::Float64, sample_size::Int
)::Vector{Float64}
    if low > high
        throw(ArgumentError(string(
            "low (", low, ") should not be greater than high ", high
        )))
    end
    validate_mixture_args(components)
    if low >= high
        throw(ArgumentError("low is greater or equal to high, low: $(low), high: $(high)"))
    end
    m = mus(components)
    s = sigmas(components)
    w = weights(components)
    d = Distributions.truncated(
        Distributions.MixtureModel(Distributions.Normal.(m, s), w),
        low, high,
    )
    return rand(d, sample_size)
end
"""
$(TYPEDSIGNATURES)
GMM1 without low, high or q
"""
function GMM1(components::Vector{DistDetails}, sample_size::Int)::Vector{Float64}
    validate_mixture_args(components)
    m = mus(components)
    s = sigmas(components)
    w = weights(components)
    d = Distributions.MixtureModel(Distributions.Normal.(m, s), w)
    return rand(d, sample_size)
end
"""
$(TYPEDSIGNATURES)
GMM1 with low, high and q
"""
function GMM1(
    components::Vector{DistDetails}, low::Float64, high::Float64, q::Float64,
    sample_size::Int
)::Vector{Float64}
    samples = GMM1(components, low, high, sample_size)

    return round.(samples ./ q) .* q
end
"""
$(TYPEDSIGNATURES)
GMM1 with q
"""
function GMM1(
    components::Vector{DistDetails}, q::Float64, sample_size::Int
)::Vector{Float64}
    samples = GMM1(components, sample_size)

    return round.(samples ./ q) .* q
end

function logprob(
    samples::Vector{Float64}, components::Vector{DistDetails}, low::Float64, high::Float64,
    q::Float64, p_accept::Float64
)::Vector{Float64}
    m = mus(components)
    s = sigmas(components)
    w = weights(components)
    d = Distributions.MixtureModel(Distributions.Normal.(m, s), w)
    ubound = min.(samples .+ (q / 2.0), high)
    lbound = max.(samples .- (q / 2.0), low)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end
function logprob(
    samples::Vector{Float64}, components::Vector{DistDetails}, q::Float64, p_accept::Float64
)::Vector{Float64}
    m = mus(components)
    s = sigmas(components)
    w = weights(components)
    d = Distributions.MixtureModel(Distributions.Normal.(m, s), w)
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
    samples::Vector{Float64}, components::Vector{DistDetails}, p_accept::Float64
)::Vector{Float64}
    m = mus(components)
    s = sigmas(components)
    w = weights(components)
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
    samples::Vector{Float64}, components::Vector{DistDetails}, low::Float64, high::Float64,
    q::Float64
)::Vector{Float64}
    isempty(samples) && return []
    validate_mixture_args(components)
    p_accept = sum(
        weights(components) .* (
            normal_cdf([high], components) - normal_cdf([low], components)
        )
    )
    return logprob(samples, components, low, high, q, p_accept)
end
"""
$(TYPEDSIGNATURES)
GMM1_lpdf with q
"""
function GMM1_lpdf(
    samples::Vector{Float64}, components::Vector{DistDetails}, q::Float64
)::Vector{Float64}
    isempty(samples) && return []
    validate_mixture_args(components)
    p_accept = 1.0

    return logprob(samples, components, q, p_accept)
end
"""
$(TYPEDSIGNATURES)
GMM1_lpdf with low, high
"""
function GMM1_lpdf(
    samples::Vector{Float64}, components::Vector{DistDetails}, low::Float64, high::Float64
)::Vector{Float64}
    isempty(samples) && return []
    validate_mixture_args(components)
    p_accept = sum(
        weights(components) .* (
            normal_cdf([high], components) - normal_cdf([low], components)
        )
    )

    return mahal(samples, components, p_accept)
end
"""
$(TYPEDSIGNATURES)
GMM1_lpdf without low, high or q
"""
function GMM1_lpdf(samples::Vector{Float64}, components::Vector{DistDetails})::Vector{Float64}
    isempty(samples) && return []
    validate_mixture_args(components)

    return mahal(samples, components, 1.0)
end

end # module GMM
