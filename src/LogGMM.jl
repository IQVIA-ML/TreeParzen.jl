module LogGMM

import Distributions
using DocStringExtensions

import ..GMM

function lgmm_samples(draws::Vector{Float64}, sample_size::Int)::Matrix{Float64}
    return reshape(exp.(draws), (sample_size, 1))
end

function quantise_samples(samples::Matrix{Float64}, q::Float64)::Matrix{Float64}
    return round.(samples ./ q) .* q
end

"""
$(TYPEDSIGNATURES)

LGMM1 with low, high
"""
function LGMM1(
    components::Vector{GMM.DistDetails}, low::Float64, high::Float64, sample_size::Int
)::Matrix{Float64}
    return lgmm_samples(
        GMM.GMM1(components, low, high, sample_size), sample_size
    )
end
"""
$(TYPEDSIGNATURES)

LGMM1 without low, high
"""
function LGMM1(components::Vector{GMM.DistDetails}, sample_size::Int)::Matrix{Float64}
    return lgmm_samples(GMM.GMM1(components, sample_size), sample_size)
end
"""
$(TYPEDSIGNATURES)

LGMM1 with low, high and q
"""
function LGMM1(
    components::Vector{GMM.DistDetails}, low::Float64, high::Float64, q::Float64,
    sample_size::Int
)::Matrix{Float64}
    samples = LGMM1(components, low, high, sample_size)
    return quantise_samples(samples, q)
end
"""
$(TYPEDSIGNATURES)

LGMM1 without low or high
"""
function LGMM1(
    components::Vector{GMM.DistDetails}, q::Real, sample_size::Int
)::Matrix{Float64}
    samples = LGMM1(components, sample_size)
    return quantise_samples(samples, Float64(q))
end

function logprob(
    samples::Matrix{Float64}, components::Vector{GMM.DistDetails}, low::Real, high::Real,
    q::Real, p_accept::Real
)::Matrix{Float64}
    mus = [c.mu for c in components]
    sigmas = [c.sigma for c in components]
    weights = [c.weight for c in components]
    d = Distributions.MixtureModel(Distributions.LogNormal.(mus, sigmas), weights)
    ubound = min.(samples .+ (q / 2.0), exp(high))
    lbound = max.(samples .- (q / 2.0), exp(low))
    lbound = max.(lbound, 0)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end
function logprob(
    samples::Matrix{Float64}, components::Vector{GMM.DistDetails}, q::Real, p_accept::Real
)::Matrix{Float64}
    mus = [c.mu for c in components]
    sigmas = [c.sigma for c in components]
    weights = [c.weight for c in components]
    d = Distributions.MixtureModel(Distributions.LogNormal.(mus, sigmas), weights)
    ubound = samples .+ (q / 2.0)
    lbound = max.(samples .- (q / 2.0), 0)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end


"""
$(TYPEDSIGNATURES)

Log-density of each row of `samples` under a mixture of lognormals (`components`).

Matches Hyperopt `tpe.LGMM1_lpdf` with `q is None`: log-space 'low' and 'high' bounds are not in this return value
(truncation is when drawing, e.g. `LGMM1` / `Samplers.loguniform`); with bounds and quantisation use
`LGMM1_lpdf(..., low, high, q)`.
"""
function LGMM1_lpdf(
    samples::Matrix{Float64}, components::Vector{GMM.DistDetails}
)::Vector{Float64}
    isempty(samples) && return Float64[]
    GMM.validate_mixture_args(components)
    x = samples[:]
    # log-normal mixture log pdf = Gaussian mixture log pdf at log(x) minus log(x) (Jacobian).
    return GMM.mahal(log.(x), components, 1.0) .- log.(x)
end
function LGMM1_lpdf(
    samples::Matrix{Float64}, components::Vector{GMM.DistDetails}, q::Float64
)::Matrix{Float64}
    isempty(samples) && return zeros(size(samples))
    GMM.validate_mixture_args(components)
    return logprob(samples, components, q, 1.0)
end
function LGMM1_lpdf(
    samples::Matrix{Float64}, components::Vector{GMM.DistDetails}, low::Float64, high::Float64,
    q::Float64
)::Matrix{Float64}
    isempty(samples) && return zeros(size(samples))
    GMM.validate_mixture_args(components)
    weights = [c.weight for c in components]
    p_accept = sum(
        weights .* (GMM.normal_cdf([high], components) - GMM.normal_cdf([low], components))
    )
    return logprob(samples, components, low, high, q, p_accept)
end

end # module LogGMM
