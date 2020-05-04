"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogUniform <: AbstractDistDelayed
    low::NestedFloat
    high::NestedFloat
end

function loguniform(low::Float64, high::Float64)::Float64
    return exp(uniform(low, high))
end
