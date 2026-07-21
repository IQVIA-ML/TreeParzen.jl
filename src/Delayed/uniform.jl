"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Uniform{L, H} <: AbstractDistDelayed
    low::L
    high::H
end

function uniform(low::Float64, high::Float64)::Float64
    return only(rand(Distributions.Uniform(low, high), 1))
end
