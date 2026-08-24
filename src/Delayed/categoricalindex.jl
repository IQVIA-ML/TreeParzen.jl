"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct CategoricalIndex <: AbstractDistDelayed
    probabilities::Vector{Float64}
end

function categoricalindex(
    probabilities::Probabilities, sample_size::Int
)::IndexObjects.IndexVector
    iszero(sample_size) && return IndexObjects.IndexVector(Int[])

    sample = transpose(rand(Distributions.Multinomial(1, probabilities.v), sample_size))
    rval = sample * (1:length(probabilities))

    return IndexObjects.IndexVector(rval)
end

categoricalindex(probabilities::AbstractVector{<:Real}, sample_size::Int) =
    categoricalindex(Probabilities(probabilities), sample_size)
