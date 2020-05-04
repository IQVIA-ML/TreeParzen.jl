"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Uniform <: AbstractDistDelayed
    low::NestedFloat
    high::NestedFloat
end

function uniform(low::Float64, high::Float64)::Float64
    return first(rand(Distributions.Uniform(low, high), 1))
end
