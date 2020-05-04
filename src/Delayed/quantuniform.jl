"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct QuantUniform <: AbstractDistDelayed
    low::NestedFloat
    high::NestedFloat
    q::NestedFloat
end

function quantuniform(low::Float64, high::Float64, q::Float64)::Float64
    return round(uniform(low, high) / q) * q
end
