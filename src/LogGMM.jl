module LogGMM

import Distributions
using DocStringExtensions
import SpecialFunctions

import ..GMM

"""
$(TYPEDSIGNATURES)

Formula copied from wikipedia
https://en.wikipedia.org/wiki/Log-normal_distribution
"""
function lognormal_lpdf(
    x::Matrix{Float64}, mu::Vector{Float64}, sigma::Vector{Float64}
)::Matrix{Float64}
    if all(sigma .< 0)
        throw(DimensionMismatch("sigma: $(sigma) is below 0"))
    end

    sigma = transpose(max.(sigma, eps(Float64)))

    Z = x * sigma .* sqrt(2 * pi)
    if size(Z) != (length(x), length(sigma))
        throw(DimensionMismatch("size(Z): $(size(Z)) does not equal $((length(x), length(sigma)))"))
    end
    E = 1//2 .* ((log.(x) .- transpose(mu)) ./ sigma) .^ 2
    result = -E .- log.(Z)
    if size(result) != (length(x), length(sigma))
        throw(DimensionMismatch("size(result): $(size(result)) does not equal$((length(x), length(sigma)))"))
    end

    return result
end

"""
$(TYPEDSIGNATURES)

LGMM1 with low, high
"""
function LGMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    low::Float64, high::Float64, sample_size::Int
)::Matrix{Float64}
    if low > high
        throw(ArgumentError(string(
            "low (", low, ") should not be greater than high ", high
        )))
    end
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end

    sample_size_t = (sample_size, 1)

    n_samples = Int(prod(sample_size_t))
    # -- draw from truncated components
    samples = Float64[]
    while length(samples) < n_samples
        active_real::Real = argmax(rand(Distributions.Multinomial(1, weights)))
        draw = rand(Distributions.Normal(mus[active_real], sigmas[active_real]))
        if low <= draw < high
            push!(samples, exp(draw))
        end
    end

    return reshape(samples, sample_size_t)
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
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end

    sample_size_t = (sample_size, 1)

    n_samples = Int(prod(sample_size_t))
    active_arr::Array = getindex.(
        permutedims(argmax(rand(Distributions.Multinomial(1, weights), n_samples); dims = 1)),
        1
    )
    if length(active_arr) != n_samples
        throw(DimensionMismatch(string(
            "length(active_arr): ", length(active_arr),
            " doesn't equal n_samples: ", n_samples
        )))
    end
    samples = exp.(rand.(Distributions.Normal.(mus[active_arr], sigmas[active_arr])))

    return reshape(samples, sample_size_t)
end
"""
$(TYPEDSIGNATURES)

LGMM1 with low, high and q
"""
function LGMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    low::Float64, high::Float64, q::Float64, sample_size::Int
)::Matrix{Float64}
    samples = LGMM1(weights, mus, sigmas, low, high, sample_size)

    return round.(samples ./ q) .* q
end
"""
$(TYPEDSIGNATURES)

LGMM1 without low or high
"""
function LGMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64}, q::Real,
    sample_size::Int
)::Matrix{Float64}
    samples = LGMM1(weights, mus, sigmas, sample_size)

    return round.(samples ./ q) .* q
end

# LGMM1_lpdf

"""
$(TYPEDSIGNATURES)

Implements CDF as given by `0.5 + 0.5 erf( log(x) - mu / sqrt(2 sigma^2))`

The maximum is used to move negative values and `0` up to a point where they do not cause
`NaN` or `Inf`, but also don't contribute much to the CDF.
"""
function lognormal_cdf(
    x::Matrix{<: Real}, mu::Vector{Float64}, sigma::Vector{Float64}
)::Matrix{Float64}
    if length(x) != length(mu) && length(x) != 1 && length(mu) != 1
        throw(DimensionMismatch("x: $(x) and mu $(mu) are not the same length or 1"))
    end

    top = log.(max.(x, eps(Float64))) .- mu
    bottom = max.(sqrt(2) .* sigma, eps(Float64))
    z = top ./ bottom
    return 0.5 .+ (0.5 .* SpecialFunctions.erf.(z))
end

function logprob(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64},
    low::Real, high::Real, q::Real, p_accept::Real
)::Matrix{Float64}
    prob = zeros(size(samples))
    for (w, mu, sigma) in zip(weights, mus, sigmas)
        ubound = min.(samples .+ (q / 2.0), exp(high))
        lbound = max.(samples .- (q / 2.0), exp(low))
        lbound = max.(0, lbound)
        # -- two-stage addition is slightly more numerically accurate
        inc_amt = w .* lognormal_cdf(ubound, [mu], [sigma])
        inc_amt .-= w .* lognormal_cdf(lbound, [mu], [sigma])
        prob += inc_amt
    end

    return log.(prob) .- log(p_accept)
end
function logprob(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64},
    q::Real, p_accept::Real
)::Matrix{Float64}
    prob = zeros(size(samples))
    for (w, mu, sigma) in zip(weights, mus, sigmas)
        ubound = samples .+ (q / 2.0)
        lbound = samples .- (q / 2.0)
        lbound = max.(0, lbound)
        # -- two-stage addition is slightly more numerically accurate
        inc_amt = w .* lognormal_cdf(ubound, [mu], [sigma])
        inc_amt .-= w .* lognormal_cdf(lbound, [mu], [sigma])
        prob += inc_amt
    end

    return log.(prob) .- log(p_accept)
end


"""
$(TYPEDSIGNATURES)

LGMM1_lpdf without low, high
(LGMM1_lpdf with low, high was identical)
"""
function LGMM1_lpdf(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}
)::Vector{Float64}
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end

    # compute the lpdf of each sample under each component
    lpdfs = lognormal_lpdf(samples, mus, sigmas)
    rval = GMM.logsum_rows(lpdfs .+ transpose(log.(weights)))
    if length(rval) != size(samples, 1)
        throw(DimensionMismatch(string(
            "Length of rval (", length(rval), ") does not match size of samples (",
            size(samples), ")"
        )))
    end

    return rval
end

function LGMM1_lpdf(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, low::Float64, high::Float64
)::Vector{Float64}
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end

    # compute the lpdf of each sample under each component
    lpdfs = lognormal_lpdf(samples, mus, sigmas)
    rval = GMM.logsum_rows(lpdfs .+ transpose(log.(weights)))
    if length(rval) != size(samples, 1)
        throw(DimensionMismatch(string(
            "Length of rval (", length(rval), ") does not match size of samples (",
            size(samples), ")"
        )))
    end

    return rval
end
function LGMM1_lpdf(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, q::Float64
)::Matrix{Float64}
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end

    p_accept = 1
    return logprob(samples, weights, mus, sigmas, q, p_accept)
end
function LGMM1_lpdf(
    samples::Matrix{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, low::Float64, high::Float64, q::Float64
)::Matrix{Float64}
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end

    p_accept = sum(weights .* (GMM.normal_cdf([high], mus, sigmas) - GMM.normal_cdf([low], mus, sigmas)))
    return logprob(samples, weights, mus, sigmas, low, high, q, p_accept)
end

end # module LogGMM
