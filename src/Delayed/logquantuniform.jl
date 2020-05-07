"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogQuantUniform <: AbstractDistDelayed
    low::NestedFloat
    high::NestedFloat
    q::NestedFloat
end

function logquantuniform(low::Float64, high::Float64, q::Float64)::Float64
    return exp(round(uniform(low, high)) / q) * q
end
