"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct QuantNormal{M, S, Q} <: AbstractDistDelayed
    mu::M
    sigma::S
    q::Q
end

function quantnormal(mu::Float64, sigma::Float64, q::Float64)::Float64
    return round(normal(mu, sigma) / q) * q
end
