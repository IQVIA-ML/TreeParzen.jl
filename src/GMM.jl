"""
Gaussian Mixture Models
"""
module GMM

import Distributions
using DocStringExtensions
import SpecialFunctions

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
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas)
        )))
    end
    if low >= high
        throw(ArgumentError("low is greater or equal to high, low: $(low), high: $(high)"))
    end
    samples = Float64[]
    while length(samples) < sample_size
        active = argmax(rand(Distributions.Multinomial(1, weights)))
        draw = rand(Distributions.Normal(mus[active], sigmas[active]))
        if low <= draw < high
            push!(samples, draw)
        end
    end

    return samples
end
"""
$(TYPEDSIGNATURES)
GMM1 without low, high or q
"""
function GMM1(
    weights::Vector{Float64}, mus::Vector{Float64}, sigmas::Vector{Float64},
    sample_size::Int
)::Vector{Float64}

    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end
    active_arr = getindex.(
        permutedims(
            argmax(rand(Distributions.Multinomial(1, weights), sample_size); dims = 1)
        ),
        1
    )
    samples = rand.(Distributions.Normal.(mus[active_arr], sigmas[active_arr]))

    return reshape(samples, sample_size)
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

# GMM1_lpdf

function logprob(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, low::Float64, high::Float64, q::Float64, p_accept::Float64
)::Vector{Float64}
    prob = zeros(size(samples))
    for (w, mu, sigma) in zip(weights, mus, sigmas)
        ubound = min.(samples .+ (q / 2.0), high)
        lbound = max.(samples .- (q / 2.0), low)
        # -- two-stage addition is slightly more numerically accurate
        inc_amt = w .* normal_cdf(ubound, [mu], [sigma])
        inc_amt .-= w .* normal_cdf(lbound, [mu], [sigma])
        prob += inc_amt
    end

    return log.(prob) .- log(p_accept)
end
function logprob(
    samples::Vector{Float64}, weights::Vector{Float64}, mus::Vector{Float64},
    sigmas::Vector{Float64}, q::Float64, p_accept::Float64
)::Vector{Float64}
    prob = zeros(size(samples))
    for (w, mu, sigma) in zip(weights, mus, sigmas)
        ubound = samples .+ (q / 2.0)
        lbound = samples .- (q / 2.0)
        # -- two-stage addition is slightly more numerically accurate
        inc_amt = w .* normal_cdf(ubound, [mu], [sigma])
        inc_amt .-= w .* normal_cdf(lbound, [mu], [sigma])
        prob += inc_amt
    end

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
    dist = reshape(samples, length(samples), 1) .- reshape(mus, 1, length(mus))
    # mahal size is (n_samples, n_components)
    mahal = (dist ./ reshape(max.(sigmas, eps(Float64)), 1, length(sigmas))) .^ 2
    Z = sqrt.(2pi .* (sigmas .^ 2))
    coef = weights ./ Z ./ p_accept

    return logsum_rows(-0.5 .* mahal .+ log.(reshape(coef, 1, length(coef))))
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
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end
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
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end
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
    if !(length(weights) == length(mus) == length(sigmas))
        throw(DimensionMismatch(string(
            "length(weights): ", length(weights),
            " doesn't equal length(mus): ", length(mus),
            " nor length(sigmas): ", length(sigmas),
        )))
    end

    return mahal(samples, weights, mus, sigmas, 1.0)
end

end # module GMM
