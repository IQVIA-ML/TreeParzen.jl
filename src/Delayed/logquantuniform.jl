"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogQuantUniform <: LogUniformQuantDist
    low::NestedFloat
    high::NestedFloat
    q::NestedFloat
end

function logquantuniform(low::Float64, high::Float64, q::Float64)::Float64
    return exp(quantuniform(low, high, q))
end

loguniformquant(x::LogQuantUniform, low, high, q) = logquantuniform(low, high, q)
