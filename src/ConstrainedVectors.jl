"""
Single-field wrappers that keep semantically distinct float vectors from being
confused with each other.

These follow the same pattern as `IndexObjects` (and the Julia discourse guidance
on restrictive type aliases): an immutable struct around a concrete `Vector` /
`Matrix` with validation in the constructor. There is essentially no runtime
cost when field types are concrete.

"""
module ConstrainedVectors

using DocStringExtensions

export Observations, PosteriorDraws, LogPosteriorDraws, ObsWeights, Probabilities, ObsPair

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Historical hyperparameter values for one parameter id (the TPE “observations”
fed into adaptive Parzen / categorical modelling). Distinct from candidate
posterior draws and from mixture component means.
"""
struct Observations
    v::Vector{Float64}

    Observations(v::Vector{Float64}) = new(copy(v))
end

Observations() = Observations(Float64[])
Observations(v::AbstractVector{<:Real}) = Observations(collect(Float64, v))

Base.isempty(o::Observations) = isempty(o.v)
Base.length(o::Observations) = length(o.v)
Base.eltype(::Type{Observations}) = Float64
Base.iterate(o::Observations, state...) = iterate(o.v, state...)
Base.getindex(o::Observations, i) = o.v[i]

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Candidate samples drawn from the *below* (good) model, scored by expected improvement against the
above model. Linear-space GMM draws.
"""
struct PosteriorDraws
    v::Vector{Float64}

    PosteriorDraws(v::Vector{Float64}) = new(copy(v))
end

PosteriorDraws() = PosteriorDraws(Float64[])
PosteriorDraws(v::AbstractVector{<:Real}) = PosteriorDraws(collect(Float64, v))

Base.isempty(o::PosteriorDraws) = isempty(o.v)
Base.length(o::PosteriorDraws) = length(o.v)
Base.eltype(::Type{PosteriorDraws}) = Float64
Base.iterate(o::PosteriorDraws, state...) = iterate(o.v, state...)
Base.getindex(o::PosteriorDraws, i) = o.v[i]
Base.minimum(o::PosteriorDraws) = minimum(o.v)
Base.maximum(o::PosteriorDraws) = maximum(o.v)

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Log-space candidate samples from LGMM (`n_draws × 1` matrix).
"""
struct LogPosteriorDraws
    v::Matrix{Float64}

    function LogPosteriorDraws(v::Matrix{Float64})
        size(v, 2) == 1 || throw(DimensionMismatch(
            "LogPosteriorDraws must have one column, got size $(size(v))"
        ))
        return new(copy(v))
    end
end

LogPosteriorDraws(v::AbstractMatrix{<:Real}) = LogPosteriorDraws(Matrix{Float64}(v))

Base.isempty(o::LogPosteriorDraws) = isempty(o.v)
Base.size(o::LogPosteriorDraws) = size(o.v)
Base.length(o::LogPosteriorDraws) = length(o.v)
Base.getindex(o::LogPosteriorDraws, inds...) = getindex(o.v, inds...)
Base.iterate(o::LogPosteriorDraws, state...) = iterate(o.v, state...)

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Per-observation importance weights in `[0, 1]` (e.g. linear forgetting).
Not a probability simplex — values need not sum to 1.
"""
struct ObsWeights
    v::Vector{Float64}

    function ObsWeights(v::Vector{Float64})
        if any(x -> x < 0 || x > 1, v)
            throw(DomainError(v, "ObsWeights elements must lie in [0, 1]"))
        end
        return new(copy(v))
    end
end

ObsWeights(v::AbstractVector{<:Real}) = ObsWeights(collect(Float64, v))

Base.isempty(o::ObsWeights) = isempty(o.v)
Base.length(o::ObsWeights) = length(o.v)

"""
$(TYPEDEF)
$(TYPEDFIELDS)

A categorical probability simplex: non-negative entries that sum to 1
(within floating-point tolerance). Used for priors and for normalised
posterior category weights (historically misnamed “pseudocounts”).
"""
struct Probabilities
    v::Vector{Float64}

    function Probabilities(v::Vector{Float64})
        if isempty(v)
            return new(copy(v))
        end
        if any(<(0), v)
            throw(DomainError(v, "Probabilities must be non-negative"))
        end
        s = sum(v)
        if !isapprox(s, 1.0; atol = 1e-8)
            throw(DomainError(v, "Probabilities must sum to 1, got $(s)"))
        end
        return new(copy(v))
    end
end

Probabilities(v::AbstractVector{<:Real}) = Probabilities(collect(Float64, v))

Base.isempty(o::Probabilities) = isempty(o.v)
Base.length(o::Probabilities) = length(o.v)
Base.getindex(o::Probabilities, i) = o.v[i]
Base.iterate(o::Probabilities, state...) = iterate(o.v, state...)

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Paired below/above observation buckets. Keeps the good and bad TPE sets from
being swapped at call sites (identical element types otherwise).
"""
struct ObsPair{T}
    below::T
    above::T
end

end # module ConstrainedVectors
