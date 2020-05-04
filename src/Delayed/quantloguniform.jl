"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct QuantLogUniform <: AbstractDistDelayed
    low::NestedFloat
    high::NestedFloat
    q::NestedFloat
end

function quantloguniform(low::Float64, high::Float64, q::Float64)::Float64
    return round(exp(uniform(low, high)) / q) * q
end
