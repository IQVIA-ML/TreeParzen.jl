"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct QuantUniform{L, H, Q} <: AbstractDistDelayed
    low::L
    high::H
    q::Q
end

function quantuniform(low::Float64, high::Float64, q::Float64)::Float64
    return round(uniform(low, high) / q) * q
end
