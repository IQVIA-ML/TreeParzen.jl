module LogGMM

import Distributions
using DocStringExtensions

import ..GMM

sample_shape(sample_size::Int) = (sample_size, 1)

function lgmm_samples(draws::Vector{Float64}, sample_size::Int)::Matrix{Float64}
    return reshape(exp.(draws), sample_shape(sample_size))
end

function quantise_samples(samples::Matrix{Float64}, q::Float64)::Matrix{Float64}
    return round.(samples ./ q) .* q
end

"""
$(TYPEDSIGNATURES)

LGMM1 with low, high
"""
function LGMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    low::Float64, high::Float64, sample_size::Int
)::Matrix{Float64}
    return lgmm_samples(
        GMM.GMM1(weights, mus, sigmas, low, high, sample_size), sample_size
    )
end
"""
$(TYPEDSIGNATURES)

LGMM1 without low, high
"""
function LGMM1(
    # weights must be Floats or Multinomial will fail
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    sample_size::Int
)::Matrix{Float64}
    return lgmm_samples(GMM.GMM1(weights, mus, sigmas, sample_size), sample_size)
end
"""
$(TYPEDSIGNATURES)

LGMM1 with low, high and q
"""
function LGMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    low::Float64, high::Float64, q::Float64, sample_size::Int
)::Matrix{Float64}
    samples = lgmm_samples(
        GMM.GMM1(weights, mus, sigmas, low, high, sample_size), sample_size
    )
    return quantise_samples(samples, q)
end
"""
$(TYPEDSIGNATURES)

LGMM1 without low or high
"""
function LGMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64}, q::Real,
    sample_size::Int
)::Matrix{Float64}
    samples = lgmm_samples(
        GMM.GMM1(weights, mus, sigmas, sample_size), sample_size
    )
    return quantise_samples(samples, Float64(q))
end

function logprob(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64},
    low::Real, high::Real, q::Real, p_accept::Real
)::Matrix{Float64}
    d = Distributions.MixtureModel(Distributions.LogNormal.(mus, sigmas), weights)
    ubound = min.(samples .+ (q / 2.0), exp(high))
    lbound = max.(samples .- (q / 2.0), exp(low))
    lbound = max.(lbound, 0)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end
function logprob(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64},
    q::Real, p_accept::Real
)::Matrix{Float64}
    d = Distributions.MixtureModel(Distributions.LogNormal.(mus, sigmas), weights)
    ubound = samples .+ (q / 2.0)
    lbound = max.(samples .- (q / 2.0), 0)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end


"""
$(TYPEDSIGNATURES)

Log-density of each row of `samples` under a mixture of lognormals (`weights`, `mus`, `sigmas`).

Matches Hyperopt `tpe.LGMM1_lpdf` with `q is None`: log-space 'low' and 'high' bounds are not in this return value
(truncation is when drawing, e.g. `LGMM1` / `Samplers.loguniform`); with bounds and quantisation use
`LGMM1_lpdf(..., low, high, q)`.
"""
function LGMM1_lpdf(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}
)::Vector{Float64}
    isempty(samples) && return Float64[]
    GMM.validate_mixture_args(weights, mus, sigmas)
    x = samples[:]
    # log-normal mixture log pdf = Gaussian mixture log pdf at log(x) minus log(x) (Jacobian).
    return GMM.mahal(log.(x), weights, mus, sigmas, 1.0) .- log.(x)
end
function LGMM1_lpdf(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, q::Float64
)::Matrix{Float64}
    isempty(samples) && return zeros(size(samples))
    GMM.validate_mixture_args(weights, mus, sigmas)
    return logprob(samples, weights, mus, sigmas, q, 1.0)
end
function LGMM1_lpdf(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, low::Float64, high::Float64, q::Float64
)::Matrix{Float64}
    isempty(samples) && return zeros(size(samples))
    GMM.validate_mixture_args(weights, mus, sigmas)
    p_accept = sum(
        weights .* (GMM.normal_cdf([high], mus, sigmas) - GMM.normal_cdf([low], mus, sigmas))
    )
    return logprob(samples, weights, mus, sigmas, low, high, q, p_accept)
end

end # module LogGMM
