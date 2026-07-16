"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogQuantUniform{L, H, Q} <: LogUniformQuantDist
    low::L
    high::H
    q::Q
end

function logquantuniform(low::Float64, high::Float64, q::Float64)::Float64
    return exp(quantuniform(low, high, q))
end

loguniformquant(x::LogQuantUniform, low, high, q) = logquantuniform(low, high, q)
