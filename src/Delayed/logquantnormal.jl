"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogQuantNormal <: AbstractDistDelayed
    mu::NestedFloat
    sigma::NestedFloat
    q::NestedFloat
end

function logquantnormal(mu::Float64, sigma::Float64, q::Float64)::Float64
    return exp(round(normal(mu, sigma)) / q) * q
end
