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
    mixture::GMM.DistDetails, low::Float64, high::Float64, sample_size::Int
)::Matrix{Float64}
    return lgmm_samples(
        GMM.GMM1(mixture, low, high, sample_size), sample_size
    )
end
"""
$(TYPEDSIGNATURES)

LGMM1 without low, high
"""
function LGMM1(mixture::GMM.DistDetails, sample_size::Int)::Matrix{Float64}
    return lgmm_samples(GMM.GMM1(mixture, sample_size), sample_size)
end
"""
$(TYPEDSIGNATURES)

LGMM1 with low, high and q
"""
function LGMM1(
    mixture::GMM.DistDetails, low::Float64, high::Float64, q::Float64, sample_size::Int
)::Matrix{Float64}
    samples = LGMM1(mixture, low, high, sample_size)
    return quantise_samples(samples, q)
end
"""
$(TYPEDSIGNATURES)

LGMM1 without low or high
"""
function LGMM1(mixture::GMM.DistDetails, q::Real, sample_size::Int)::Matrix{Float64}
    samples = LGMM1(mixture, sample_size)
    return quantise_samples(samples, Float64(q))
end

function logprob(
    samples::Matrix{Float64}, mixture::GMM.DistDetails, low::Real, high::Real, q::Real,
    p_accept::Real
)::Matrix{Float64}
    d = Distributions.MixtureModel(
        Distributions.LogNormal.(mixture.mus, mixture.sigmas), mixture.weights
    )
    ubound = min.(samples .+ (q / 2.0), exp(high))
    lbound = max.(samples .- (q / 2.0), exp(low))
    lbound = max.(lbound, 0)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end
function logprob(
    samples::Matrix{Float64}, mixture::GMM.DistDetails, q::Real, p_accept::Real
)::Matrix{Float64}
    d = Distributions.MixtureModel(
        Distributions.LogNormal.(mixture.mus, mixture.sigmas), mixture.weights
    )
    ubound = samples .+ (q / 2.0)
    lbound = max.(samples .- (q / 2.0), 0)
    prob = Distributions.cdf.(Ref(d), ubound) .- Distributions.cdf.(Ref(d), lbound)
    return log.(prob) .- log(p_accept)
end


"""
$(TYPEDSIGNATURES)

Log-density of each row of `samples` under a mixture of lognormals (`mixture`).

Matches Hyperopt `tpe.LGMM1_lpdf` with `q is None`: log-space 'low' and 'high' bounds are not in this return value
(truncation is when drawing, e.g. `LGMM1` / `Samplers.loguniform`); with bounds and quantisation use
`LGMM1_lpdf(..., low, high, q)`.
"""
function LGMM1_lpdf(samples::Matrix{Float64}, mixture::GMM.DistDetails)::Vector{Float64}
    isempty(samples) && return Float64[]
    x = samples[:]
    # log-normal mixture log pdf = Gaussian mixture log pdf at log(x) minus log(x) (Jacobian).
    return GMM.mahal(log.(x), mixture, 1.0) .- log.(x)
end
function LGMM1_lpdf(
    samples::Matrix{Float64}, mixture::GMM.DistDetails, q::Float64
)::Matrix{Float64}
    isempty(samples) && return zeros(size(samples))
    return logprob(samples, mixture, q, 1.0)
end
function LGMM1_lpdf(
    samples::Matrix{Float64}, mixture::GMM.DistDetails, low::Float64, high::Float64, q::Float64
)::Matrix{Float64}
    isempty(samples) && return zeros(size(samples))
    p_accept = sum(
        mixture.weights .* (
            GMM.normal_cdf([high], mixture) - GMM.normal_cdf([low], mixture)
        )
    )
    return logprob(samples, mixture, low, high, q, p_accept)
end

end # module LogGMM
