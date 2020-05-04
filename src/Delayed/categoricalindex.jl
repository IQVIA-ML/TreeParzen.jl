"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct CategoricalIndex <: AbstractDistDelayed
    probabilities::Vector{Float64}
end

function categoricalindex(
    probabilities::Vector{Float64}, sample_size::Int
)::IndexObjects.IndexVector
    iszero(sample_size) && return []

    sample = transpose(rand(Distributions.Multinomial(1, probabilities), sample_size))
    rval = sample * (1:length(probabilities))

    return IndexObjects.IndexVector(rval)
end
