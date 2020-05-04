"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct QuantNormal <: AbstractDistDelayed
    mu::NestedFloat
    sigma::NestedFloat
    q::NestedFloat
end

function quantnormal(mu::Float64, sigma::Float64, q::Float64)::Float64
    return round(normal(mu, sigma) / q) * q
end
