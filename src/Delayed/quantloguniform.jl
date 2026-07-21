"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct QuantLogUniform{L, H, Q} <: LogUniformQuantDist
    low::L
    high::H
    q::Q
end

function quantloguniform(low::Float64, high::Float64, q::Float64)::Float64
    return round(exp(uniform(low, high)) / q) * q
end

loguniformquant(x::QuantLogUniform, low, high, q) = quantloguniform(low, high, q)
